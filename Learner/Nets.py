from typing import Tuple

import torch
import torch as T
import torch.nn as nn
import torch_geometric as pyg

from Environment import Game
from .Utils import extract_attr, sparse_face_matrix, preprocess_adj, sparse_misc_node, get_masks
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


def nn_sum(tensor: T.Tensor) -> T.Tensor:
    return tensor.permute(0, 3, 1, 2).sum(dim=-2, keepdim=True).sum(-1, keepdim=True).permute(0, 2, 3, 1)


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

        # TODO: Fix Magic Numbers
        self.node_adj = T.eye(74, dtype=T.long).unsqueeze(0)
        self.node_adj[:, 54:, 54:] = 0
        self.edge_adj = pyg.utils.to_dense_adj(self.sparse_edge, max_num_nodes=74)
        self.face_adj = pyg.utils.to_dense_adj(self.sparse_face, max_num_nodes=74)
        self.face_adj = self.face_adj + T.eye(self.face_adj.shape[-1])
        self.face_adj[0, self.node_adj[0]] = 0
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

        self.node_mask = self.node_mask.flatten()
        self.edge_mask = self.edge_mask.flatten()
        self.face_mask = self.face_mask.flatten()
        self.full_mask = self.full_mask.flatten()

        self.face_adj_norm = preprocess_adj(self.face_adj, batch_size, add_self_loops=False)
        self.edge_adj_norm = preprocess_adj(self.edge_adj, batch_size, add_self_loops=False)
        self.full_adj_norm = preprocess_adj(self.full_adj, batch_size, add_self_loops=False)

        self.node_embed = MLP(n_node_attr + n_players * n_player_attr + 2, n_embed, residual=False)
        self.edge_embed = MLP(n_edge_attr + n_players * n_player_attr + 2, n_embed, residual=False)
        self.face_embed = MLP(n_face_attr, n_embed, residual=False)

        self.action_value = MLP(n_embed, n_output, final=True)
        self.state_value = MLP(n_embed, n_output, final=True)

        self.power_layers = nn.Sequential(*[

            PowerfulLayer(n_embed, n_embed, self.full_adj_norm.to(self.on_device)).to(self.on_device)
            for _ in range(n_power_layers)

        ])

        self.state_matrix = self.state_matrix.to(self.on_device)
        self.full_mask = self.full_mask.to(self.on_device)
        self.out_matrix = self.out_matrix.to(self.on_device)

        if load_state:
            self.load()

    def forward(self, observation: T.Tensor) -> T.Tensor:
        observation = observation.to(self.on_device)
        mask = ~self.action_mask.unsqueeze(-1).repeat(observation.shape[0], 1, 1, self.n_output).to(self.on_device)

        obs_matrix = self.feature_embedding(observation)
        obs_matrix = self.power_layers(obs_matrix)

        action_matrix = self.action_value(obs_matrix)
        state_matrix = self.state_value(obs_matrix)

        action_matrix[mask] = 0
        state_matrix[mask] = 0

        mean_action = nn_sum(action_matrix) / nn_sum(mask)

        q_matrix = action_matrix + (state_matrix - mean_action)
        q_matrix[mask] = -T.inf
        return q_matrix

    def feature_embedding(self, observation):
        batch = observation.shape[0]
        f = observation.shape[-1]

        observation = observation.permute(0, 3, 1, 2).reshape((batch, f, -1))
        obs_matrix = (self.state_matrix.clone()
                      .repeat((batch, 1, 1, 1))
                      .reshape((batch, self.n_embed, -1))
                      .permute(0, 2, 1))

        node_f = observation[:, :, self.node_mask].permute(0, 2, 1)
        edge_f = observation[:, :, self.edge_mask].permute(0, 2, 1)
        face_f = observation[:, :6, self.face_mask].permute(0, 2, 1)

        obs_matrix[:, self.node_mask, :] = self.node_embed(node_f)
        obs_matrix[:, self.edge_mask, :] = self.edge_embed(edge_f)
        obs_matrix[:, self.face_mask, :] = self.face_embed(face_f)

        obs_matrix = (obs_matrix
                      .permute(0, 2, 1)
                      .reshape((batch, 74, 74, self.n_embed)))

        return obs_matrix

    def clone_state(self, other):
        self.load_state_dict(other.state_dict())

    def save(self):
        torch.save(self.state_dict(), f'./q_net_state.pth')

    def load(self):
        self.load_state_dict(torch.load(f'./q_net_state.pth'))

    def get_dense(self, game: Game) -> Tuple[T.Tensor, int]:
        node_x, edge_x, face_x = extract_attr(game)
        p0_mask = self.mask_util(game, 0).squeeze()
        p1_mask = self.mask_util(game, 1).squeeze()
        p_mask = T.stack((p0_mask, p1_mask), dim=-1).unsqueeze(0)
        mask = T.zeros(1, 74, 74, 2)
        mask[:,:54,:54,:] = p_mask
        mask[:, -1, -1, :] = True

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

    def mask_util(self, game: Game, player) -> T.Tensor:
        road_mask, village_mask = get_masks(game, player)

        if game.first_turn:
            if game.first_turn_village_switch:
                road_mask = T.zeros((self.sparse_edge.shape[1],), dtype=T.bool)
            else:
                village_mask = T.zeros((54,), dtype=T.bool)

        village_mask = T.diag_embed(village_mask).bool()
        if road_mask.sum() == 0:
            road_mask = T.zeros_like(village_mask, dtype=T.bool)
        else:
            road_mask = pyg.utils.to_dense_adj(self.sparse_edge[:, road_mask],
                                               max_num_nodes=village_mask.shape[-1]).bool()
        return village_mask + road_mask