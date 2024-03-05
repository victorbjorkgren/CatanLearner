from typing import Tuple, List

import torch
import torch as T
import torch.nn as nn
import torch_geometric as pyg
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from Environment import Game
from .Utils import sparse_face_matrix, preprocess_adj, sparse_misc_node
from .Layers import MLP, PowerfulLayer, MultiHeadAttention


class PlayerNet(nn.Module):
    def __init__(self, in_features, out_features, n_embed, n_heads):
        super(PlayerNet, self).__init__()
        self.player_embedding = MLP(in_features, n_embed)

        self.q = MLP(n_embed, n_embed // n_heads, final=True)
        self.k = MLP(n_embed, n_embed // n_heads, final=True)
        self.v = MLP(n_embed, n_embed // n_heads, final=True)

        self.multi_head_attention = MultiHeadAttention(n_embed, n_heads)

        self.output = nn.Sequential(
            MLP(n_embed, n_embed),
            MLP(n_embed, out_features, final=True)
        )

    def forward(self, player_states, board_embedding):
        player_states = self.player_embedding(player_states)
        board_embedding = board_embedding[T.argwhere(T.nonzero(board_embedding))]

        states = T.cat((player_states, board_embedding))
        del player_states, board_embedding

        q = self.q(states)
        k = self.k(states)
        v = self.v(states)

        # Multi-head attention
        attn_output = self.multi_head_attention(v, k, q)

        # Final output
        output = self.output(attn_output)
        return output


def nn_sum(tensor: T.Tensor, dims: List[int]) -> T.Tensor:
    for dim in dims:
        tensor = tensor.sum(dim, keepdim=True)
    return tensor
    # return tensor.permute(0, 3, 1, 2).sum(dim=-2, keepdim=True).sum(-1, keepdim=True).permute(0, 2, 3, 1)


class GameNet(nn.Module):
    def __init__(self,
                 game,
                 n_embed,
                 n_output,
                 n_power_layers,
                 on_device,
                 load_state,
                 batch_size=1,
                 undirected_faces=True,
                 ):
        super(GameNet, self).__init__()

        sparse_edge = game.board.state.edge_index.clone()
        sparse_face = game.board.state.face_index.clone()
        n_node_attr = game.board.state.num_node_features
        n_edge_attr = game.board.state.num_node_features
        n_face_attr = game.board.state.face_attr.shape[1]
        n_player_attr = game.players[0].state.shape[0]
        n_players = len(game.players)

        self.on_device = on_device
        self.n_output = n_output
        self.n_embed = n_embed
        self.undirected_faces = undirected_faces
        self.sparse_edge = sparse_edge
        self.sparse_face = sparse_face_matrix(sparse_face, to_undirected=self.undirected_faces)
        self.sparse_pass_node = sparse_misc_node(sparse_edge.max(), self.sparse_face.max() + 1, to_undirected=True)
        self.sparse_full = T.cat((self.sparse_edge, self.sparse_face, self.sparse_pass_node), dim=1)

        # TODO: Remove Magic Numbers
        self.node_adj = T.eye(74, dtype=T.long).unsqueeze(0)
        self.node_adj[:, 54:, 54:] = 0
        self.node_adj[:, -1, -1] = 1
        self.edge_adj = pyg.utils.to_dense_adj(self.sparse_edge, max_num_nodes=74)
        self.face_adj = pyg.utils.to_dense_adj(self.sparse_face, max_num_nodes=74)
        self.face_adj = self.face_adj + T.eye(self.face_adj.shape[-1])
        self.face_adj[self.node_adj > 0] = 0
        self.face_adj[:, -1, -1] = 0
        self.full_adj = pyg.utils.to_dense_adj(self.sparse_full)
        self.full_adj = self.full_adj + T.eye(self.full_adj.shape[-1])
        self.full_mask = self.full_adj > 0

        self.state_matrix = T.zeros(
            (self.full_mask.shape[0], self.full_mask.shape[1], self.full_mask.shape[2], self.n_embed))
        self.out_matrix = T.zeros(
            (self.state_matrix.shape[0], self.state_matrix.shape[1], self.state_matrix.shape[2], self.n_output))
        self.node_mask = self.node_adj > 0
        self.edge_mask = self.edge_adj > 0
        self.face_mask = self.face_adj > 0
        self.action_mask = (self.node_adj > 0) | (self.edge_adj > 0)
        self.action_mask[:, -1, -1] = True

        self.n_possible_actions = self.action_mask.sum()

        self.node_mask = self.node_mask.view((1, -1)).squeeze()
        self.edge_mask = self.edge_mask.view((1, -1)).squeeze()
        self.face_mask = self.face_mask.view((1, -1)).squeeze()
        self.full_mask = self.full_mask.view((1, -1)).squeeze()

        self.face_adj_norm = preprocess_adj(self.face_adj, batch_size, add_self_loops=False)
        self.edge_adj_norm = preprocess_adj(self.edge_adj, batch_size, add_self_loops=False)
        self.full_adj_norm = preprocess_adj(self.full_adj, batch_size, add_self_loops=False)

        self.node_embed = MLP(n_node_attr + n_players * n_player_attr + 2, n_embed, residual=False)
        self.edge_embed = MLP(n_edge_attr + n_players * n_player_attr + 2, n_embed, residual=False)
        self.face_embed = MLP(n_face_attr, n_embed, residual=False)

        self.action_value = MLP(n_embed, n_output, final=True)
        self.state_value = MLP(n_embed, n_output, final=True)

        self.lstm = nn.LSTM(n_embed, n_embed, batch_first=True)

        self.power_layers = nn.Sequential(*[

            PowerfulLayer(n_embed, n_embed, self.full_adj_norm.to(self.on_device)).to(self.on_device)
            for _ in range(n_power_layers)

        ])

        self.state_matrix = self.state_matrix.to(self.on_device)
        self.full_mask = self.full_mask.to(self.on_device)
        self.out_matrix = self.out_matrix.to(self.on_device)

        if load_state:
            self.load()

    def forward(self,
                observation: T.Tensor,
                seq_lengths: T.Tensor,
                h_in: T.Tensor,
                c_in: T.Tensor
                ) -> Tuple[T.Tensor, T.Tensor, T.Tensor]:

        assert len(observation.shape) == 5, "QNet wants [B, T, N, N, F]"

        obs_matrix = observation.to(self.on_device)
        h_in = h_in.to(self.on_device)
        c_in = c_in.to(self.on_device)

        obs_matrix = self.feature_embedding(obs_matrix)
        obs_matrix = self.power_layers(obs_matrix)
        obs_matrix, hn, cn = self.temporal_layer(obs_matrix, seq_lengths, h_in, c_in)
        action_matrix, state_matrix = self.action_value_heads(obs_matrix)

        mean_action = nn_sum(action_matrix, [2, 3]) / self.n_possible_actions
        q_matrix = action_matrix + (state_matrix - mean_action)

        # Mask invalid actions
        b, s, n, _, f = q_matrix.shape
        neg_mask = ~self.action_mask[None, :, :, :, None].repeat(b, s, 1, 1, f)
        q_matrix[neg_mask] = -T.inf
        return q_matrix, hn, cn

    def temporal_layer(
            self,
            obs_matrix: T.Tensor,
            seq_lengths: T.Tensor,
            h_in: T.Tensor,
            c_in: T.Tensor
    ) -> Tuple[T.Tensor, T.Tensor, T.Tensor]:

        packed_matrix = pack_padded_sequence(
            obs_matrix[:, :, -1, -1, :],
            seq_lengths,
            batch_first=True,
            enforce_sorted=False
        )
        packed_matrix, (hn, cn) = self.lstm(packed_matrix, (h_in, c_in))
        temporal_matrix, _ = pad_packed_sequence(packed_matrix, batch_first=True)

        # Add temporal matrix to all elements of observation matrix
        obs_matrix[:, :temporal_matrix.shape[1]] = (

                temporal_matrix[:, :, None, None, :]
                + obs_matrix[:, :temporal_matrix.shape[1]]

        )

        return obs_matrix, hn, cn

    def feature_embedding(self, observation):
        batch, seq, n, _, f = observation.shape

        x_flat = observation.view(batch * seq, n * n, f)
        obs_matrix = T.zeros((batch * seq, n * n, self.n_embed), dtype=T.float).to(self.on_device)

        obs_matrix[:, self.node_mask] = self.node_embed(x_flat[:, self.node_mask])
        obs_matrix[:, self.edge_mask] = self.edge_embed(x_flat[:, self.edge_mask])
        obs_matrix[:, self.face_mask] = self.face_embed(x_flat[:, self.face_mask, :6])

        obs_matrix = obs_matrix.reshape(batch, seq, n, n, self.n_embed)

        return obs_matrix

    def action_value_heads(self, observation):
        batch, seq, n, _, f = observation.shape

        x_flat = observation.view(batch * seq, n * n, f)
        action_matrix = T.zeros((batch * seq, n * n, self.n_output), dtype=T.float).to(self.on_device)
        state_matrix = T.zeros((batch * seq, n * n, self.n_output), dtype=T.float).to(self.on_device)

        action_matrix[:, self.full_mask] = self.action_value(x_flat[:, self.full_mask])
        state_matrix[:, self.full_mask] = self.state_value(x_flat[:, self.full_mask])

        action_matrix = action_matrix.reshape(batch, seq, n, n, self.n_output)
        state_matrix = state_matrix.reshape(batch, seq, n, n, self.n_output)

        return action_matrix, state_matrix

    def clone_state(self, other):
        self.load_state_dict(other.state_dict())

    def save(self):
        torch.save(self.state_dict(), f'./q_net_state.pth')

    def load(self):
        self.load_state_dict(torch.load(f'./q_net_state.pth'))

    def get_dense(self, game: Game) -> Tuple[T.Tensor, int]:
        node_x, edge_x, face_x = self.extract_attr(game)
        p0_mask = self.mask_util(game, 0).long()  # .squeeze()
        p1_mask = self.mask_util(game, 1).long()  # .squeeze()
        # p_mask = T.stack((p0_mask, p1_mask), dim=-1).unsqueeze(0)
        mask = T.zeros(1, 74, 74, 2)
        mask[:, p0_mask[0, :], p0_mask[1, :], 0] = 1
        mask[:, p1_mask[0, :], p1_mask[1, :], 1] = 1
        mask[:, -1, -1, :] = 1

        # Normalize-ish
        node_x /= 10
        edge_x /= 10
        face_x /= 12

        pass_x = T.zeros_like(node_x)
        # TODO: Make board size and n_player invariant
        face_x = T.cat((face_x, T.zeros(19, 8)), dim=1)
        node_x = T.cat((node_x, face_x, T.zeros((1, 14))))
        node_matrix = T.diag_embed(node_x.permute(1, 0)).permute(1, 2, 0).unsqueeze(0)
        face_x = face_x.repeat_interleave(6, 0)
        if self.undirected_faces:
            face_x = T.cat((face_x, face_x.flip(0)), dim=0)
        connection_x = T.cat((edge_x, face_x, pass_x, pass_x))
        connection_matrix = pyg.utils.to_dense_adj(self.sparse_full, edge_attr=connection_x)

        full_matrix = node_matrix + connection_matrix
        full_matrix = T.cat((full_matrix, mask), dim=-1)
        return full_matrix, game.current_player

    @staticmethod
    def extract_attr(game: Game):
        node_x = game.board.state.x.clone()
        edge_x = game.board.state.edge_attr.clone()
        face_x = game.board.state.face_attr.clone()

        player_states = T.cat([ps.state.clone()[None, :] for ps in game.players], dim=1)
        node_x = T.cat((node_x, player_states.repeat((node_x.shape[0], 1))), dim=1)
        edge_x = T.cat((edge_x, player_states.repeat((edge_x.shape[0], 1))), dim=1)

        return node_x, edge_x, face_x

    @staticmethod
    def mask_util(game: Game, player) -> T.Tensor:
        # road_mask, village_mask = get_dense_masks(game, player)
        road_mask = game.board.sparse_road_mask(player, game.players[player].hand, game.first_turn)
        village_mask = game.board.sparse_village_mask(player, game.players[player].hand, game.first_turn)

        mask = T.cat((road_mask, village_mask.repeat(2, 1)), dim=1)
        return mask

        # if game.first_turn:
        #     if game.first_turn_village_switch:
        #         road_mask = T.zeros((self.sparse_edge.shape[1],), dtype=T.bool)
        #     else:
        #         village_mask = T.zeros((54,), dtype=T.bool)
        #
        # village_mask = T.diag_embed(village_mask).bool()
        # if road_mask.sum() == 0:
        #     road_mask = T.zeros_like(village_mask, dtype=T.bool)
        # else:
        #     road_mask = pyg.utils.to_dense_adj(self.sparse_edge[:, road_mask],
        #                                        max_num_nodes=village_mask.shape[-1]).bool()
        # return village_mask + road_mask
