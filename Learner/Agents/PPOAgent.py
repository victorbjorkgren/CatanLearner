from collections import deque, namedtuple
from random import randint

import torch
import torch as T

from Environment import Game
from Environment.constants import N_NODES, N_ROADS
from Learner.Agents.BaseAgent import BaseAgent
from Learner.Utility.ActionTypes import BaseAction, TradeAction, NoopAction, BuildAction, \
    sparse_type_mapping, RoadAction, SettlementAction, SparsePi
from Learner.Utility.CustomDistributions import SparseCatanActionSampler
from Learner.Utility.DataTypes import PPOActionPack, PPORewardPack, PPOTransition, GameState, NetInput
from Learner.Nets import PPONet
from Learner.PrioReplayBuffer import PrioReplayBuffer
from Learner.Utility.Utils import TensorUtils
from Learner.constants import N_PLAYERS


class PPOAgent(BaseAgent):
    def __init__(
            self,
            net: PPONet,
            game: Game,
            buffer: PrioReplayBuffer,
            max_sequence_length: int,
            burn_in_length: int,
            is_titan: bool = False,
            history_display: int = 100
    ) -> None:
        super().__init__(game, buffer, max_sequence_length, burn_in_length, is_titan, history_display)
        self.name = "PPOAgent"
        self.net = net

        self.action_pack_buffer = deque(maxlen=max_sequence_length)
        self.game_state_buffer = deque(maxlen=max_sequence_length)
        self.reward_buffer = deque(maxlen=max_sequence_length)

        self.ticks_since_flush = self.max_seq_length - self.burn_in_length

        self.h0 = T.zeros((1, 1, 32), dtype=T.float)
        self.c0 = T.zeros((1, 1, 32), dtype=T.float)

    def sample_action(self, state: GameState, i_am_player: int) -> BaseAction:
        # state_key = TensorUtils.get_cache_key(state)

        if self.my_name in ['Titan', 'latest']:
            self.game_state_buffer.append(state)

        ht, ct = self.h0, self.c0

        # Get masks
        buildable = self.get_sparse_build_mask(i_am_player)
        road_mask = torch.zeros((N_ROADS * 2))
        settle_mask = torch.zeros((N_NODES))
        road_mask[buildable.road_mask] = 1.0
        settle_mask[buildable.village_mask] = 1.0
        # build_mask = torch.zeros((N_NODES, N_NODES))
        # build_mask[buildable[0, :], buildable[1, :]] = 1
        # build_mask = ~state[0, :54, :54, -N_PLAYERS+i_am_player].bool()
        tradable = self.game.can_trade(i_am_player, 4)
        trade_mask = T.isin(T.arange(5), tradable)
        no_op_mask = self.game.can_no_op()

        with T.no_grad():
            net_out: PPONet.Output = self.net(
                NetInput(state, T.Tensor([1]), self.h0, self.c0)
            )
            self.h0, self.c0 = net_out.hn, net_out.cn

        pi = self.net.get_pi(net_out, i_am_player)

        # Apply masks
        pi.settlement *= settle_mask
        pi.road *= road_mask
        pi.trade.give *= trade_mask

        type_mask = torch.ones_like(pi.type)
        type_mask[:, :, sparse_type_mapping.inverse[NoopAction]] *= no_op_mask
        if pi.trade.give.nonzero().numel() == 0:
            trade_type_ind = sparse_type_mapping.inverse[TradeAction]
            type_mask[:, :, trade_type_ind] = 0
            pi.trade.give += 1 / pi.trade.give.numel()
        if pi.road.nonzero().numel() == 0:
            road_type_ind = sparse_type_mapping.inverse[RoadAction]
            type_mask[:, :, road_type_ind] = 0
            pi.road += 1 / pi.road.numel()
        if pi.settlement.nonzero().numel() == 0:
            settle_type_ind = sparse_type_mapping.inverse[SettlementAction]
            type_mask[:, :, settle_type_ind] = 0
            pi.settlement += 1 / pi.settlement.numel()

        pi.type *= type_mask

        # Re-normalize probs
        pi.settlement = pi.settlement / pi.settlement.sum(-1)
        pi.road = pi.road / pi.road.sum(-1)
        pi.trade.give = pi.trade.give / pi.trade.give.sum(-1)
        pi.type = pi.type / pi.type.sum(-1)

        # Sample
        sampler = SparseCatanActionSampler(pi)
        action = sampler.sample()

        if self.my_name in ['Titan', 'latest']:
            logprob = sampler.log_prob(action)[None, None, :]
            value = net_out.state_value[:1, :1, i_am_player]
            masks = SparsePi(
                type=type_mask,
                road=road_mask[None, None, :],
                settlement=settle_mask[None, None, :],
                trade=TradeAction(give=trade_mask[None, None, :],
                                  get=torch.ones_like(pi.trade.get))
            )
            self.action_pack_buffer.append(PPOActionPack(action, masks, logprob, value, ht, ct))
        return action

    def get_build_mask(self, player) -> T.Tensor:
        road_mask = self.game.board.sparse_road_mask(player, self.game.players[player].hand, self.game.first_turn, self.game.first_turn_village_switch)
        village_mask = self.game.board.sparse_village_mask(player, self.game.players[player].hand, self.game.first_turn, self.game.first_turn_village_switch)

        mask = T.cat((road_mask, village_mask.repeat(2, 1)), dim=1)
        return mask

    def get_sparse_build_mask(self, player) -> namedtuple('sparse_mask', ['road_mask', 'village_mask']):
        road_mask = self.game.board.sparse_road_mask(player, self.game.players[player].hand, self.game.first_turn, self.game.first_turn_village_switch)
        village_mask = self.game.board.sparse_village_mask(player, self.game.players[player].hand, self.game.first_turn, self.game.first_turn_village_switch)
        if road_mask.numel() > 0:
            _, road_inds = TensorUtils.pairwise_isin(road_mask, self.game.board.state.edge_index)
            road_mask = road_inds[:, 1]
        return namedtuple('sparse_mask', ['road_mask', 'village_mask'])(road_mask, village_mask)

    def signal_episode_done(self, i_am_player: int) -> None:
        super().signal_episode_done(i_am_player)
        self.action_pack_buffer.clear()
        self.game_state_buffer.clear()
        self.reward_buffer.clear()
        self.h0 *= 0.
        self.c0 *= 0.

    def clear_cache(self):
        pass

    def _flush_buffer(self, done: bool, i_am_player: int, force: bool = False) -> None:
        if self.my_name not in ['Titan', 'latest']:
            return
        assert len(self.action_pack_buffer) == len(self.reward_buffer) == len(self.game_state_buffer)
        has_length = len(self.reward_buffer) >= (self.max_seq_length - self.burn_in_length)
        time_to_flush = self.ticks_since_flush > 0
        if (has_length & time_to_flush) | done:
            transition = PPOTransition(
                state=GameState.concat(list(self.game_state_buffer), dim=1),
                seq_lens=torch.tensor([len(self.reward_buffer)]),
                action_pack=PPOActionPack.concat(list(self.action_pack_buffer), dim=1),
                reward_pack=PPORewardPack.concat(list(self.reward_buffer), dim=1)
            )
            self.buffer.add(transition)
            expected_length = self.max_seq_length - self.burn_in_length
            self.ticks_since_flush = randint(-expected_length, -(expected_length // 3))

    def update_score(self, reward):
        if reward > 0.:
            self.episode_score += 1

    def update_reward(self, reward: float, done: bool, game: Game, i_am_player: int) -> None:
        if self.my_name not in ['Titan', 'latest']:
            return
        rew_pack = PPORewardPack(
            T.tensor([[reward]]),
            T.tensor([[game.episode]]),
            T.tensor([[done]]),
            T.tensor([[i_am_player]])
        )
        self.reward_buffer.append(rew_pack)
        self.ticks_since_flush += 1

        self._flush_buffer(done, i_am_player, force=False)

