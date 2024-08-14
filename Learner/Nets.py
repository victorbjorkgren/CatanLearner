import os
from abc import abstractmethod
from collections import namedtuple
from datetime import datetime
from typing import Tuple

import torch
import torch as T
import torch.nn as nn
import torch_geometric as pyg
from torch import Tensor
from torch_geometric.nn import GATv2Conv

from Environment.constants import N_RESOURCES, N_GRAPH_NODES, N_NODES, N_ROADS
from Learner.Layers import MLP, PowerfulLayer, MultiHeadAttention
from Learner.Utility.ActionTypes import TradeAction, sparse_type_mapping, SparsePi, FlatPi
from Learner.Utility.DataTypes import PPOTransition, NetInput, GameState, NetOutput
from Learner.Utility.Utils import TensorUtils, Holders
from Learner.constants import DENSE_FORWARD


# file_lock = multiprocessing.Lock()


# def synchronized_state(func):
#     def state_wrapper(*args, **kwargs):
#         with file_lock:
#             return func(*args, **kwargs)
#     return state_wrapper


class BaseNet(nn.Module):
    def __init__(self, name, lock):
        super().__init__()
        self.name = name
        self.lock = lock

    def clone_state(self, other):
        self.load_state_dict(other.state_dict(), strict=False)

    # @synchronized_state
    def save(self, suffix):
        with self.lock:
            if suffix == 'latest':
                torch.save(self.state_dict(), f'./{self.name}_Agent_Titan.pth')
            else:
                os.makedirs('./PastTitans/', exist_ok=True)
                torch.save(self.state_dict(), f'./PastTitans/{self.name}_Agent_{suffix}.pth')

                # Create BackUp
                now = datetime.now()
                now = now.strftime("%Y%m%d_%H%M%S")
                os.makedirs('./NetBackup/', exist_ok=True)
                torch.save(self.state_dict(), f'./NetBackup/{self.name}_state_{now}.pth')

    # @synchronized_state
    def load(self, file: str | int):
        with self.lock:
            if file == 'latest':
                dir = './'
                file = f'{self.name}_Agent_Titan.pth'
            else:
                dir = './PastTitans'
            path = os.path.join(dir, file)
            try:
                self.load_state_dict(torch.load(path), strict=False)
            except FileNotFoundError:
                print(f'Could not find {file} - saving with fresh weights')
                torch.save(self.state_dict(), path)



class CoreNet(BaseNet):
    # keys = ['action_type', 'action_road', 'action_settle', 'action_trade', 'state_value', 'hn', 'cn']
    keys = ['action_index', 'state_value', 'hn', 'cn']
    Output = namedtuple('Output', keys)

    def __init__(self,
                 net_config: dict,
                 batch_size=1,
                 undirected_faces=True, ) -> None:
        super().__init__('Core', net_config['lock'])
        game = net_config['game']
        n_embed = net_config['n_embed']
        n_output = net_config['n_output']
        n_power_layers = net_config['n_power_layers']

        sparse_edge = game.board.state.edge_index.clone()
        sparse_face = game.board.state.face_index.clone()
        n_node_attr = game.board.state.num_node_features
        n_edge_attr = game.board.state.num_node_features
        n_face_attr = game.board.state.face_attr.shape[1]
        n_game_attr = game.num_game_attr
        # n_player_attr = game.players[0].state.shape[0]
        self.n_players = len(game.players)

        self.on_device = net_config['on_device']
        self.n_output = n_output
        self.n_embed = n_embed
        self.undirected_faces = undirected_faces
        self.sparse_edge = sparse_edge
        self.sparse_face = TensorUtils.sparse_face_matrix(sparse_face, to_undirected=self.undirected_faces)
        self.sparse_game_node = TensorUtils.sparse_game_node(
            sparse_edge.max(),
            self.sparse_face.max() + 1,
            to_undirected=True
        )
        self.sparse_full = T.cat((self.sparse_edge, self.sparse_face, self.sparse_game_node), dim=1)

        self.node_adj = T.eye(N_GRAPH_NODES, dtype=T.long).unsqueeze(0)
        self.node_adj[:, N_NODES:, N_NODES:] = 0
        self.edge_adj = pyg.utils.to_dense_adj(self.sparse_edge, max_num_nodes=N_GRAPH_NODES)
        self.face_node_adj = T.eye(N_GRAPH_NODES, dtype=T.long).unsqueeze(0)
        self.face_node_adj[self.node_adj > 0] = 0
        self.face_node_adj[:, -1, -1] = 0
        self.face_adj = pyg.utils.to_dense_adj(self.sparse_face, max_num_nodes=N_GRAPH_NODES)
        self.face_adj = self.face_adj + self.face_node_adj
        self.game_node_adj = T.zeros_like(self.node_adj, dtype=T.long)
        self.game_node_adj[:, -1, -1] = 1
        self.game_adj = pyg.utils.to_dense_adj(self.sparse_game_node, max_num_nodes=N_GRAPH_NODES)
        self.game_adj = self.game_adj + self.game_node_adj
        self.full_adj = pyg.utils.to_dense_adj(self.sparse_full)
        self.full_adj = self.full_adj + T.eye(self.full_adj.shape[-1])
        self.full_mask = self.full_adj > 0

        # self.sparse_full = pyg.utils.add_self_loops(self.sparse_full)[0]
        self.state_matrix = T.zeros(
            (self.full_mask.shape[0], self.full_mask.shape[1], self.full_mask.shape[2], self.n_embed))
        self.out_matrix = T.zeros(
            (self.state_matrix.shape[0], self.state_matrix.shape[1], self.state_matrix.shape[2], self.n_output))
        self.node_mask = self.node_adj > 0
        self.edge_mask = self.edge_adj > 0
        self.face_mask = self.face_node_adj > 0
        self.game_mask = self.game_node_adj > 0
        self.action_mask = (self.node_adj > 0) | (self.edge_adj > 0)
        self.action_mask[:, -1, -1] = True

        self.n_possible_actions = self.action_mask.sum()

        self.node_mask = self.node_mask.view((1, -1)).squeeze()
        self.edge_mask = self.edge_mask.view((1, -1)).squeeze()
        self.face_mask = self.face_mask.view((1, -1)).squeeze()
        self.game_mask = self.game_mask.view((1, -1)).squeeze()
        self.full_mask = self.full_mask.view((1, -1)).squeeze()

        self.face_adj_norm = TensorUtils.preprocess_adj(self.face_adj, batch_size, add_self_loops=False)
        self.edge_adj_norm = TensorUtils.preprocess_adj(self.edge_adj, batch_size, add_self_loops=False)
        self.full_adj_norm = TensorUtils.preprocess_adj(self.full_adj, batch_size, add_self_loops=False)

        self.node_embed = MLP(self.n_players * 2 + N_RESOURCES, n_embed, residual=False)
        self.edge_embed = MLP(self.n_players * 2, n_embed, residual=False)
        self.face_embed = MLP(n_face_attr, n_embed, residual=False)
        self.game_embed = MLP(n_game_attr, n_embed, residual=False)

        self.action_value = MLP(n_embed, n_output, activated_out=False)
        self.state_value = MLP(n_embed, n_output, activated_out=False)

        self.action_trade_give = MLP(n_embed, N_RESOURCES * self.n_players, activated_out=False)
        self.action_trade_get = MLP(n_embed, N_RESOURCES * self.n_players, activated_out=False)
        self.state_trade_give = MLP(n_embed, N_RESOURCES * self.n_players, activated_out=False)
        self.state_trade_get = MLP(n_embed, N_RESOURCES * self.n_players, activated_out=False)

        self.lstm = nn.LSTM(n_embed, n_embed, batch_first=True)

        if DENSE_FORWARD:
            self.forward_func = self.dense_forward
            self.power_layers = nn.Sequential(*[
                PowerfulLayer(n_embed, n_embed, self.full_adj_norm.to(self.on_device)).to(self.on_device)
                for _ in range(n_power_layers)
            ])
            # self.n_action_types = len(dense_type_mapping)
            raise NotImplementedError
        else:
            self.forward_func = self.flat_forward
            self.noop_head = MLP(n_embed, 1, activated_out=False)
            n_heads = 2
            assert n_embed % n_heads == 0
            self.gnn_layers = nn.ModuleList([GATv2Conv(n_embed, n_embed, edge_dim=n_embed, n_heads=n_heads) for _ in range(n_power_layers)])
            # self.gnn_layers = nn.ModuleList([GINEConv(MLP(n_embed, n_embed), train_eps=True) for _ in range(n_power_layers)])
            # self.gnn_layers = nn.ModuleList([GCNConv()])
            self.n_action_types = len(sparse_type_mapping)
            self.edge_padding = torch.zeros(self.sparse_full.shape[1], n_embed)

        # self.action_type_head = MLP(n_embed, self.n_action_types * self.n_players, activated_out=False)

        self.state_matrix = self.state_matrix.to(self.on_device)
        self.full_mask = self.full_mask.to(self.on_device)
        self.out_matrix = self.out_matrix.to(self.on_device)

    def forward(self, core_input: NetInput) -> Output:
        return self.forward_func(core_input)

    def dense_forward(self, core_input: NetInput) -> Output:
        observation = core_input.state
        seq_lengths = core_input.seq_lens
        h_in = core_input.lstm_h[:1, :, :]
        c_in = core_input.lstm_c[:1, :, :]
        # assert len(observation.shape) == 5, "QNet wants [B, T, N, N, F]"
        b, t = observation.node_features.shape[:2]

        obs_matrix = observation.to(self.on_device)
        del observation
        h_in = h_in.to(self.on_device)
        c_in = c_in.to(self.on_device)

        obs_matrix = self.dense_feature_embedding(obs_matrix)
        obs_matrix = self.power_layers(obs_matrix)
        obs_matrix, hn, cn = self.dense_temporal_layer(obs_matrix, seq_lengths, h_in, c_in)
        action_matrix, state_value = self.dense_action_value_heads(obs_matrix)

        game_state = obs_matrix[:, :, -1, -1, :]
        del obs_matrix
        action_trade_get = self.action_trade_get(game_state).view((b, t, N_RESOURCES, self.n_players))
        action_trade_give = self.action_trade_give(game_state).view((b, t, N_RESOURCES, self.n_players))
        # state_trade_get = self.state_trade_get(game_state).view((b, t, N_RESOURCES, self.n_players))
        # state_trade_give = self.state_trade_give(game_state).view((b, t, N_RESOURCES, self.n_players))

        action_trade = TradeAction(give=action_trade_give, get=action_trade_get)
        # action_trade = T.stack((action_trade_give, action_trade_get), dim=2)
        # state_trade = T.stack((state_trade_give, state_trade_get), dim=2)

        action_type = self.action_type_head(game_state).view((b, t, self.n_action_types, self.n_players))
        # ['action_type', 'action_matrix', 'state_matrix', 'action_trade' 'state_trade', 'hn', 'cn']
        return self.Output(action_type, action_matrix, action_trade, state_value, hn, cn)

    def sparse_forward(self, core_input: NetInput) -> Output:
        observation = core_input.state
        seq_lengths = core_input.seq_lens
        h_in = core_input.lstm_h[:1, :, :]
        c_in = core_input.lstm_c[:1, :, :]
        # assert len(observation.shape) == 5, "QNet wants [B, T, N, N, F]"
        b, t = observation.node_features.shape[:2]

        observation = observation.to(self.on_device)
        h_in = h_in.to(self.on_device)
        c_in = c_in.to(self.on_device)
        hn, cn = h_in, c_in

        node_f, edge_f = self.sparse_feature_embedding(observation)
        edge_index = torch.cat([(self.sparse_full.max() + 1) * n + self.sparse_full for n in range(b * t)], dim=1).to(self.on_device)
        src, dst = edge_index
        edge_pad = self.edge_padding.repeat(b, t, 1, 1).to(self.on_device)
        edge_pad[:, :, :N_ROADS * 2, :] = edge_f
        edge_f = edge_pad
        node_f = node_f.view(b * t * N_GRAPH_NODES, -1)
        edge_f = edge_f.view(b * t * self.sparse_full.shape[1], -1)
        for layer in self.gnn_layers:
            node_f = layer(node_f, edge_index=edge_index, edge_attr=edge_f)
            edge_f = (node_f[src][:, None, :] * node_f[dst][:, :, None]).sum(dim=-1)
        node_f = node_f.view(b, t, N_GRAPH_NODES, -1)
        edge_f = edge_f.view(b, t, self.sparse_full.shape[1], -1)
        # temporal_element, (hn, cn) = self.lstm(node_f[:, :, -1, :], (h_in, c_in))
        # node_f, hn, cn = self.dense_temporal_layer(node_f, seq_lengths, h_in, c_in)
        # edge_f = (node_f[src] * node_f[dst]).sum(dim=-1)

        state_value = self.state_value(node_f[:, :, -1, :])
        action_type, edge_action, node_action, action_trade = self.nestled_heads(node_f, edge_f)
        return self.Output(action_type, edge_action, node_action, action_trade, state_value, hn, cn)

    def flat_forward(self, core_input: NetInput) -> Output:
        observation = core_input.state
        seq_lengths = core_input.seq_lens
        h_in = core_input.lstm_h[:1, :, :]
        c_in = core_input.lstm_c[:1, :, :]
        # assert len(observation.shape) == 5, "QNet wants [B, T, N, N, F]"
        b, t = observation.node_features.shape[:2]

        observation = observation.to(self.on_device)
        h_in = h_in.to(self.on_device)
        c_in = c_in.to(self.on_device)
        hn, cn = h_in, c_in

        node_f, edge_f = self.sparse_feature_embedding(observation)
        edge_index = torch.cat([(self.sparse_full.max() + 1) * n + self.sparse_full for n in range(b * t)], dim=1).to(
            self.on_device)
        src, dst = edge_index
        edge_pad = self.edge_padding.repeat(b, t, 1, 1).to(self.on_device)
        edge_pad[:, :, :N_ROADS * 2, :] = edge_f
        edge_f = edge_pad
        node_f = node_f.view(b * t * N_GRAPH_NODES, -1)
        edge_f = edge_f.view(b * t * self.sparse_full.shape[1], -1)
        for layer in self.gnn_layers:
            node_f = layer(node_f, edge_index=edge_index, edge_attr=edge_f)
            edge_f = (node_f[src][:, None, :] * node_f[dst][:, :, None]).sum(dim=-1)
        node_f = node_f.view(b, t, N_GRAPH_NODES, -1)
        edge_f = edge_f.view(b, t, self.sparse_full.shape[1], -1)
        # temporal_element, (hn, cn) = self.lstm(node_f[:, :, -1, :], (h_in, c_in))
        # node_f, hn, cn = self.dense_temporal_layer(node_f, seq_lengths, h_in, c_in)
        # edge_f = (node_f[src] * node_f[dst]).sum(dim=-1)

        state_value = self.state_value(node_f[:, :, -1, :])
        action_pi = self.flat_heads(node_f, edge_f)
        return self.Output(action_pi, state_value, hn, cn)

    def flat_heads(self, node_f, edge_f) -> FlatPi:
        b, t = node_f.shape[:2]
        node_action = self.action_value(node_f[:, :, :N_NODES, :])
        edge_action = self.action_value(edge_f[:, :, :N_ROADS * 2, :])

        action_trade_get = self.action_trade_get(node_f[:, :, -1, :]).view((b, t, N_RESOURCES, self.n_players))
        action_trade_give = self.action_trade_give(node_f[:, :, -1, :]).view((b, t, N_RESOURCES, self.n_players))

        action_trade = TradeAction(give=action_trade_give, get=action_trade_get)

        action_noop = self.noop_head(node_f[:, :, -1, :])
        action_pi = FlatPi.stack_parts(road=edge_action, settle=node_action, trade=action_trade, noop=action_noop)
        return action_pi

    def nestled_heads(self, node_f, edge_f):
        b, t = node_f.shape[:2]
        node_action = self.action_value(node_f[:, :, :N_NODES, :])
        edge_action = self.action_value(edge_f[:, :, :N_ROADS * 2, :])

        action_trade_get = self.action_trade_get(node_f[:, :, -1, :]).view((b, t, N_RESOURCES, self.n_players))
        action_trade_give = self.action_trade_give(node_f[:, :, -1, :]).view((b, t, N_RESOURCES, self.n_players))

        action_trade = TradeAction(give=action_trade_give, get=action_trade_get)

        action_type = self.action_type_head(node_f[:, :, -1, :]).view((b, t, self.n_action_types, self.n_players))
        # ['action_type', 'action_matrix', 'state_matrix', 'action_trade' 'state_trade', 'hn', 'cn']
        return action_type, edge_action, node_action, action_trade

    def sparse_feature_embedding(self, observation: GameState):
        node_features = self.node_embed(observation.node_features)
        edge_features = self.edge_embed(observation.edge_features)
        face_features = self.face_embed(observation.face_features)
        game_features = self.game_embed(observation.game_features)
        node_f = torch.cat((
            node_features,
            face_features,
            game_features), -2)
        edge_f = edge_features
        return node_f, edge_f

    def flat_feature_embedding(self, observation: GameState):
        node_features = self.node_embed(observation.node_features)
        edge_features = self.edge_embed(observation.edge_features)
        face_features = self.face_embed(observation.face_features)
        game_features = self.game_embed(observation.game_features)

    def dense_feature_embedding(self, observation: GameState):
        b, t = observation.node_features.shape[:2]

        # x_flat = observation.view(batch * seq, n * n, f)
        obs_matrix = T.zeros((b, t, N_GRAPH_NODES * N_GRAPH_NODES, self.n_embed), dtype=T.float).to(self.on_device)

        # obs_matrix[:, self.node_mask] = self.node_embed(x_flat[:, self.node_mask])
        # obs_matrix[:, self.edge_mask] = self.edge_embed(x_flat[:, self.edge_mask])
        # obs_matrix[:, self.face_mask] = self.face_embed(x_flat[:, self.face_mask, :6])

        obs_matrix[:, :, self.node_mask] = self.node_embed(observation.node_features)
        obs_matrix[:, :, self.edge_mask] = self.edge_embed(observation.edge_features)
        obs_matrix[:, :, self.face_mask] = self.face_embed(observation.face_features)
        obs_matrix[:, :, self.game_mask] = self.game_embed(observation.game_features)

        obs_matrix = obs_matrix.reshape(b, t, N_GRAPH_NODES, N_GRAPH_NODES, self.n_embed)

        return obs_matrix

    def dense_action_value_heads(self, observation):
        batch, seq, n, _, f = observation.shape

        x_flat = observation.view(batch * seq, n * n, f)
        action_matrix = T.zeros((batch * seq, n * n, self.n_output), dtype=T.float).to(self.on_device)
        # state_value = T.zeros((batch * seq, self.n_output), dtype=T.float).to(self.on_device)

        action_matrix[:, self.full_mask] = self.action_value(x_flat[:, self.full_mask])
        state_value = self.state_value(x_flat[:, -1, :])

        action_matrix = action_matrix.reshape(batch, seq, n, n, self.n_output)
        state_value = state_value.reshape(batch, seq, self.n_output)

        return action_matrix, state_value

    def dense_temporal_layer(self,
                             obs_matrix: T.Tensor,
                             seq_lengths: T.Tensor,
                             h_in: T.Tensor,
                             c_in: T.Tensor
                             ) -> Tuple[T.Tensor, T.Tensor, T.Tensor]:
        # try:
        #     packed_matrix = pack_padded_sequence(
        #         obs_matrix[:, :, -1, -1, :],
        #         seq_lengths,
        #         batch_first=True,
        #         enforce_sorted=False
        #     )
        # except RuntimeError:
        #     breakpoint()
        #     raise RuntimeError
        temporal_element, (hn, cn) = self.lstm(obs_matrix[:, :, -1, -1, :], (h_in, c_in))
        # temporal_matrix, _ = pad_packed_sequence(packed_matrix, batch_first=True)

        # Add temporal matrix to all elements of observation matrix
        # obs_matrix[:, :temporal_matrix.shape[1]] = (
        #
        #         temporal_matrix[:, :, None, None, :]
        #         + obs_matrix[:, :temporal_matrix.shape[1]]
        #
        # )
        obs_matrix = obs_matrix + temporal_element[:, :, None, None, :]

        return obs_matrix, hn, cn

    # def get_dense(self, game: Game) -> Tuple[T.Tensor, int]:
    #     node_x, edge_x, face_x = self.extract_attr(game)
    #     # p0_mask = self.mask_util(game, 0).long()  # .squeeze()
    #     # p1_mask = self.mask_util(game, 1).long()  # .squeeze()
    #     mask = T.zeros(1, N_GRAPH_NODES, N_GRAPH_NODES, game.n_players)
    #     for i in range(game.n_players):
    #         p_mask = self.mask_util(game, i).long()
    #         mask[:, p_mask[0, :], p_mask[1, :], i] = 1
    #
    #     # mask[:, p1_mask[0, :], p1_mask[1, :], 1] = 1
    #     mask[:, -1, -1, :] = game.can_no_op()
    #
    #     # Normalize-ish
    #     node_x = node_x.log()
    #     edge_x = edge_x.log()
    #     face_x = face_x.log()
    #
    #     pass_x = T.zeros_like(node_x)
    #     # face_x = T.cat((face_x, T.zeros(19, 8)), dim=1)
    #     node_x = T.cat((node_x, face_x, T.zeros((1, 7*game.n_players))))
    #     node_matrix = T.diag_embed(node_x.permute(1, 0)).permute(1, 2, 0).unsqueeze(0)
    #     face_x = face_x.repeat_interleave(6, 0)
    #     if self.undirected_faces:
    #         face_x = T.cat((face_x, face_x.flip(0)), dim=0)
    #     connection_x = T.cat((edge_x, face_x, pass_x, pass_x))
    #     connection_matrix = pyg.utils.to_dense_adj(self.sparse_full, edge_attr=connection_x)
    #
    #     full_matrix = node_matrix + connection_matrix
    #     full_matrix = T.cat((full_matrix, mask), dim=-1)
    #     return full_matrix, game.current_player

    # @staticmethod
    # def extract_attr(game: Game):
    #     node_x = game.board.state.x.clone()
    #     edge_x = game.board.state.edge_attr.clone()
    #     face_x = game.board.state.face_attr.clone()
    #
    #     player_states = T.cat([ps.state.clone()[None, :] for ps in game.players], dim=1)
    #     node_x = T.cat((node_x, player_states.repeat((node_x.shape[0], 1))), dim=1)
    #     edge_x = T.cat((edge_x, player_states.repeat((edge_x.shape[0], 1))), dim=1)
    #
    #     return GameState(node_x, edge_x, face_x)

    # @staticmethod
    # def mask_util(game: Game, player) -> T.Tensor:
    #     # road_mask, village_mask = get_dense_masks(game, player)
    #     road_mask = game.board.sparse_road_mask(player, game.players[player].hand, game.first_turn and not game.first_turn_village_switch)
    #     village_mask = game.board.sparse_village_mask(player, game.players[player].hand, game.first_turn and game.first_turn_village_switch)
    #
    #     mask = T.cat((road_mask, village_mask.repeat(2, 1)), dim=1)
    #     return mask

class PlayerNet(BaseNet):
    def __init__(self, in_features, out_features, n_embed, n_heads):
        super().__init__('Player')
        self.player_embedding = MLP(in_features, n_embed)

        self.q = MLP(n_embed, n_embed // n_heads, activated_out=True)
        self.k = MLP(n_embed, n_embed // n_heads, activated_out=True)
        self.v = MLP(n_embed, n_embed // n_heads, activated_out=True)

        self.multi_head_attention = MultiHeadAttention(n_embed, n_heads)

        self.output = nn.Sequential(
            MLP(n_embed, n_embed),
            MLP(n_embed, out_features, activated_out=True)
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


class GameNet(BaseNet):
    def __init__(self,
                 net_config,
                 name,
                 batch_size=1,
                 undirected_faces=True,
                 ):
        super().__init__(name, net_config['lock'])
        self.config = net_config
        self.name = name
        self.batch_size = batch_size
        self.undirected_faces = undirected_faces
        self.on_device = net_config['on_device']

        self._core_net = CoreNet(net_config, batch_size, undirected_faces)

        # if self.config['load_state']:
        #     self.load('latest')

    def get_dense(self, game):
        return self._core_net.get_dense(game)

    def extract_attributes(self, game):
        return self._core_net.extract_attr(game)

    @abstractmethod
    def forward(self,
                transition: Holders
                ) -> Tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor]:

        raise NotImplementedError


class QNet(GameNet):
    keys = ['q_matrix', 'trade_matrix', 'hn', 'cn']
    Output = namedtuple('Output', keys)

    def __init__(self,
                 net_config,
                 batch_size=1,
                 undirected_faces=True,
                 ):
        super().__init__(net_config, 'Q', batch_size, undirected_faces)

    def forward(self,
                observation: T.Tensor,
                seq_lengths: T.Tensor,
                h_in: T.Tensor,
                c_in: T.Tensor
                ) -> Tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor]:
        assert len(observation.shape) == 5, "QNet wants [B, T, N, N, F]"

        core = self._core_net(observation, seq_lengths, h_in, c_in)

        mean_action = TensorUtils.nn_sum(core.action_matrix, [2, 3]) / self.n_possible_actions
        q_matrix = core.action_matrix + (core.state_matrix - mean_action)

        # Mask invalid actions
        b, s, n, _, f = q_matrix.shape
        neg_mask = ~self._core_net.action_mask[None, :, :, :, None].repeat(b, s, 1, 1, f)
        q_matrix[neg_mask] = -T.inf

        mean_trade = TensorUtils.nn_sum(core.action_trade, [2, 3]) / N_RESOURCES
        q_trade = core.action_trade + (core.state_trade - mean_trade)
        return q_matrix, q_trade, core.hn, core.cn


class PPONet(GameNet):
    def __init__(self,
                 net_config,
                 batch_size=1,
                 undirected_faces=True,
                 name='PPO',
                 softmax=True
                 ):
        super().__init__(net_config, name, batch_size, undirected_faces)
        self.softmax = softmax
        if net_config['load_state']:
            self.load('latest')

        self.to(self.on_device)

    def forward(self, transition: PPOTransition) -> NetOutput:
        core = self._core_net(transition)
        # pi = core.action_index.index.softmax(-2)
        if self.softmax:
            pi = TensorUtils.stable_softmax(core.action_index.index, dim=-2)
            return NetOutput(FlatPi(pi), core.state_value, core.hn, core.cn)
        else:
            return NetOutput(FlatPi(core.action_index.index), core.state_value, core.hn, core.cn)
        # pi_type = core.action_type.softmax(-2)
        # pi_trade = TradeAction(give=core.action_trade.give.softmax(-2), get=core.action_trade.get.softmax(-2))
        # if DENSE_FORWARD:
        #     pi_map = self.masked_softmax(core.action_matrix)
        #     raise NotImplementedError
        # else:
        #     pi_road = core.action_road.softmax(-2)
        #     pi_settle = core.action_settle.softmax(-2)
        # return NetOutput(SparsePi(pi_type, pi_settle, pi_road, pi_trade), core.state_value, core.hn, core.cn)

    def dense_softmax(self, action_logits: Tensor) -> Tensor:
        b, s, n, _, f = action_logits.shape
        action_mask = self._core_net.action_mask[None, :, :, :, None].repeat(b, s, 1, 1, f).to(action_logits.device)
        z = torch.exp(action_logits) * action_mask
        sum_z = TensorUtils.nn_sum(z, [2, 3]).clamp_min(1e-9)
        action_probs = z / sum_z
        return action_probs

    def lerp_towards(self, other: nn.Module, alpha: float):
        assert 0 <= alpha <= 1

        lerp_state = {}
        for key in self.state_dict().keys():
            if key in other.state_dict().keys():
                lerp_state[key] = alpha * other.state_dict()[key] + (1 - alpha) * self.state_dict()[key]
            else:
                raise KeyError(f'Key {key} not in {other.state_dict()}')

        self.load_state_dict(lerp_state)

    @staticmethod
    def get_pi(net_out: NetOutput, i_am_player: Tensor | int):
        if isinstance(i_am_player, Tensor):
            i_am_player = i_am_player.unsqueeze(-1).unsqueeze(-1)
            pi_type = torch.gather(net_out.pi.type, -1, i_am_player.expand(-1, -1, 4, -1)).squeeze(-1)
            # pi_map = torch.gather(net_out.pi_map, -1, i_am_player.unsqueeze(-1).expand(-1, -1, N_NODES, N_NODES, -1)).squeeze(-1)
            pi_trade = TradeAction(
                give=torch.gather(net_out.pi.trade.give, -1, i_am_player.expand(-1, -1, 5, -1)).squeeze(-1),
                get=torch.gather(net_out.pi.trade.get, -1, i_am_player.expand(-1, -1, 5, -1)).squeeze(-1),
            )
            pi_road = torch.gather(net_out.pi.road, -1, i_am_player.expand(-1, -1, 144, -1)).squeeze(-1)
            pi_settle = torch.gather(net_out.pi.settlement, -1, i_am_player.expand(-1, -1, 54, -1)).squeeze(-1)
            # if pi_map.isnan().any():
            #     breakpoint()
        elif isinstance(i_am_player, int):
            pi_type = net_out.pi.type[:1, :1, :, i_am_player]
            pi_trade = TradeAction(
                give=net_out.pi.trade.give[:1, :1, :, i_am_player],
                get=net_out.pi.trade.get[:1, :1, :, i_am_player]
            )
            pi_road = net_out.pi.road[:1, :1, :, i_am_player]
            pi_settle = net_out.pi.settlement[:1, :1, :, i_am_player]
            # pi_map = net_out.pi_map[:1, :1, :N_NODES, :N_NODES, i_am_player]
        else:
            raise TypeError
        return SparsePi(type=pi_type, trade=pi_trade, road=pi_road, settlement=pi_settle)
