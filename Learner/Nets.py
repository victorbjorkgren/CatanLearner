import os
import threading
from abc import abstractmethod
from collections import namedtuple
from datetime import datetime
from typing import Tuple

import torch
import torch as T
import torch.nn as nn
import torch_geometric as pyg
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from Environment import Game
from Environment.constants import N_RESOURCES, N_ACTION_TYPES, N_GRAPH_NODES, N_NODES
from Learner.Utility.ActionTypes import TradeAction, Pi
from Learner.Utility.DataTypes import PPOTransition, NetInput, GameState
from Learner.Utility.Utils import TensorUtils, Holders
from Learner.Layers import MLP, PowerfulLayer, MultiHeadAttention

file_lock = threading.Lock()


def synchronized(func):
    def wrapper(*args, **kwargs):
        with file_lock:
            return func(*args, **kwargs)
    return wrapper


class BaseNet(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def clone_state(self, other):
        self.load_state_dict(other.state_dict(), strict=False)

    @synchronized
    def save(self, suffix):
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

    @synchronized
    def load(self, file: str | int):
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
    keys = ['action_type', 'action_matrix', 'action_trade', 'state_value', 'hn', 'cn']
    Output = namedtuple('Output', keys)

    def __init__(self,
                 net_config: dict,
                 batch_size=1,
                 undirected_faces=True, ) -> None:
        super().__init__('Core')
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

        self.node_embed = MLP(self.n_players * 2, n_embed, residual=False)
        self.edge_embed = MLP(self.n_players * 2, n_embed, residual=False)
        self.face_embed = MLP(n_face_attr, n_embed, residual=False)
        self.game_embed = MLP(n_game_attr, n_embed, residual=False)

        self.action_value = MLP(n_embed, n_output, activated_out=True)
        self.state_value = MLP(n_embed, n_output, activated_out=True)

        self.action_trade_give = MLP(n_embed, N_RESOURCES * self.n_players, activated_out=True)
        self.action_trade_get = MLP(n_embed, N_RESOURCES * self.n_players, activated_out=True)
        self.state_trade_give = MLP(n_embed, N_RESOURCES * self.n_players, activated_out=True)
        self.state_trade_get = MLP(n_embed, N_RESOURCES * self.n_players, activated_out=True)

        self.action_type_head = MLP(n_embed, N_ACTION_TYPES * self.n_players, activated_out=True)

        self.lstm = nn.LSTM(n_embed, n_embed, batch_first=True)

        self.power_layers = nn.Sequential(*[

            PowerfulLayer(n_embed, n_embed, self.full_adj_norm.to(self.on_device)).to(self.on_device)
            for _ in range(n_power_layers)

        ])

        self.state_matrix = self.state_matrix.to(self.on_device)
        self.full_mask = self.full_mask.to(self.on_device)
        self.out_matrix = self.out_matrix.to(self.on_device)

    def forward(self, core_input: NetInput) -> Output:
        observation = core_input.state
        seq_lengths = core_input.seq_lens
        h_in = core_input.lstm_h[:1, :, :]
        c_in = core_input.lstm_c[:1, :, :]
        # assert len(observation.shape) == 5, "QNet wants [B, T, N, N, F]"
        b, t = observation.node_features.shape[:2]

        obs_matrix = observation.to(self.on_device)
        h_in = h_in.to(self.on_device)
        c_in = c_in.to(self.on_device)

        obs_matrix = self.feature_embedding(obs_matrix)
        obs_matrix = self.power_layers(obs_matrix)
        obs_matrix, hn, cn = self.temporal_layer(obs_matrix, seq_lengths, h_in, c_in)
        action_matrix, state_value = self.action_value_heads(obs_matrix)

        game_state = obs_matrix[:, :, -1, -1, :]
        action_trade_get = self.action_trade_get(game_state).view((b, t, N_RESOURCES, self.n_players))
        action_trade_give = self.action_trade_give(game_state).view((b, t, N_RESOURCES, self.n_players))
        # state_trade_get = self.state_trade_get(game_state).view((b, t, N_RESOURCES, self.n_players))
        # state_trade_give = self.state_trade_give(game_state).view((b, t, N_RESOURCES, self.n_players))

        action_trade = TradeAction(give=action_trade_give, get=action_trade_get)
        # action_trade = T.stack((action_trade_give, action_trade_get), dim=2)
        # state_trade = T.stack((state_trade_give, state_trade_get), dim=2)

        action_type = self.action_type_head(game_state).view((b, t, N_ACTION_TYPES, self.n_players))
        # ['action_type', 'action_matrix', 'state_matrix', 'action_trade' 'state_trade', 'hn', 'cn']
        return self.Output(action_type, action_matrix, action_trade, state_value, hn, cn)

    def feature_embedding(self, observation: GameState):
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

    def action_value_heads(self, observation):
        batch, seq, n, _, f = observation.shape

        x_flat = observation.view(batch * seq, n * n, f)
        action_matrix = T.zeros((batch * seq, n * n, self.n_output), dtype=T.float).to(self.on_device)
        # state_value = T.zeros((batch * seq, self.n_output), dtype=T.float).to(self.on_device)

        action_matrix[:, self.full_mask] = self.action_value(x_flat[:, self.full_mask])
        state_value = self.state_value(x_flat[:, -1, :])

        action_matrix = action_matrix.reshape(batch, seq, n, n, self.n_output)
        state_value = state_value.reshape(batch, seq, self.n_output)

        return action_matrix, state_value

    def temporal_layer(self,
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
        try:
            temporal_element, (hn, cn) = self.lstm(obs_matrix[:, :, -1, -1, :], (h_in, c_in))
        except:
            breakpoint()
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
        super().__init__(name)
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
    keys = ['pi_type', 'pi_map', 'pi_trade', 'state_value', 'hn', 'cn']
    Output = namedtuple('Output', keys)

    def __init__(self,
                 net_config,
                 batch_size=1,
                 undirected_faces=True,
                 ):
        super().__init__(net_config, 'PPO', batch_size, undirected_faces)

        if net_config['load_state']:
            self.load('latest')

    def forward(self, transition: PPOTransition) -> Output:
        core = self._core_net(transition)
        pi_type = core.action_type.softmax(-2)
        pi_map = self.masked_softmax(core.action_matrix)
        pi_trade = TradeAction(give=core.action_trade.give.softmax(-2), get=core.action_trade.get.softmax(-2))
        return self.Output(pi_type, pi_map, pi_trade, core.state_value, core.hn, core.cn)

    def masked_softmax(self, action_logits: Tensor) -> Tensor:
        b, s, n, _, f = action_logits.shape
        action_mask = self._core_net.action_mask[None, :, :, :, None].repeat(b, s, 1, 1, f)
        z = torch.exp(action_logits) * action_mask
        sum_z = TensorUtils.nn_sum(z, [2, 3])
        sum_z[sum_z == 0] = 1
        action_probs = z / sum_z

        # TODO: Correction for out of sequence states
        # action_probs[action_probs.sum(-1) == 0] = 1 / action_probs.shape[-1]

        return action_probs

    @staticmethod
    def get_pi(net_out: Output, i_am_player: Tensor | int):
        if isinstance(i_am_player, Tensor):
            i_am_player = i_am_player.unsqueeze(-1).unsqueeze(-1)
            pi_type = torch.gather(net_out.pi_type, -1, i_am_player.expand(-1, -1, 3, -1)).squeeze(-1)
            pi_map = torch.gather(net_out.pi_map, -1, i_am_player.unsqueeze(-1).expand(-1, -1, N_GRAPH_NODES, N_GRAPH_NODES, -1)).squeeze(-1)
            pi_trade = TradeAction(
                give=torch.gather(net_out.pi_trade.give, -1, i_am_player.expand(-1, -1, 5, -1)).squeeze(-1),
                get=torch.gather(net_out.pi_trade.get, -1, i_am_player.expand(-1, -1, 5, -1)).squeeze(-1),
            )
        elif isinstance(i_am_player, int):
            pi_type = net_out.pi_type[0, 0, :, i_am_player]
            pi_trade = TradeAction(
                give=net_out.pi_trade.give[0, 0, :, i_am_player],
                get=net_out.pi_trade.get[0, 0, :, i_am_player]
            )
            pi_map = net_out.pi_map[0, 0, :N_NODES, :N_NODES, i_am_player]
        else:
            raise TypeError
        return Pi(type=pi_type, trade=pi_trade, map=pi_map)
