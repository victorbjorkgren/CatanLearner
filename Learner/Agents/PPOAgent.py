from collections import deque
from random import randint

import torch
import torch as T

from Environment import Game
from Learner.Agents.BaseAgent import BaseAgent
from Learner.Utility.ActionTypes import BaseAction, TradeAction, type_mapping, NoopAction, BuildAction
from Learner.Utility.CustomDistributions import CatanActionSampler
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

    def sample_action(self, state: T.Tensor, i_am_player: int) -> BaseAction:
        state_key = TensorUtils.get_cache_key(state)

        self.game_state_buffer.append(GameState(state))

        ht, ct = self.h0, self.c0

        # Get masks
        build_mask = ~state[0, :54, :54, -N_PLAYERS+i_am_player].bool()
        tradeable = self.game.can_trade(i_am_player, 4)
        trade_mask = ~T.isin(T.arange(5), tradeable)
        no_op_mask = self.game.can_no_op()

        with T.no_grad():
            net_out: PPONet.Output = self.net(
                NetInput(state.unsqueeze(1), T.Tensor([1]), self.h0, self.c0)
            )
            self.h0, self.c0 = net_out.hn, net_out.cn

        pi_type = net_out.pi_type[0, 0, :, i_am_player]
        pi_trade = TradeAction(
            give=net_out.pi_trade.give[0, 0, :, i_am_player],
            get=net_out.pi_trade.get[0, 0, :, i_am_player]
        )
        pi_map = net_out.pi_map[0, 0, :54, :54, i_am_player]

        pi_map[build_mask] = 0
        pi_trade.give[trade_mask] = 0
        pi_type[type_mapping.inverse[NoopAction]] *= no_op_mask

        if pi_trade.give.nonzero().numel() == 0:
            trade_type_ind = type_mapping.inverse[TradeAction]
            pi_type[trade_type_ind] = 0
            pi_trade.give += 1 / pi_trade.give.numel()
        if pi_map.nonzero().numel() == 0:
            build_type_ind = type_mapping.inverse[BuildAction]
            pi_type[build_type_ind] = 0
            pi_map += 1 / pi_map.numel()
        pi_type = pi_type / pi_type.sum(-1)

        sampler = CatanActionSampler(pi_type, pi_map, pi_trade)
        action = sampler.sample()
        # if isinstance(action, BuildAction):
        #     try:
        #         self.game.decode_build_action(action)
        #     except:
        #         breakpoint()
        logprob = sampler.log_prob(action).unsqueeze(0)
        value = net_out.state_value[0, 0, i_am_player].unsqueeze(0)

        self.action_pack_buffer.append(PPOActionPack(action, logprob, value, ht, ct))
        return action

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
        assert len(self.action_pack_buffer) == len(self.reward_buffer) == len(self.game_state_buffer)
        has_length = len(self.reward_buffer) >= (self.max_seq_length - self.burn_in_length)
        time_to_flush = self.ticks_since_flush > 0
        if (has_length & time_to_flush) | done:
            transition = PPOTransition(
                state=GameState.concat(self.game_state_buffer, dim=0),
                seq_lens=torch.tensor([len(self.reward_buffer)]),
                action_pack=PPOActionPack.concat(self.action_pack_buffer, dim=0),
                reward_pack=PPORewardPack.stack(self.reward_buffer, dim=0)
            )
            self.buffer.add(transition)
            expected_length = self.max_seq_length - self.burn_in_length
            self.ticks_since_flush = randint(-expected_length, -(expected_length // 3))

    def update_reward(self, reward: float, done: bool, game: Game, i_am_player: int) -> None:
        rew_pack = PPORewardPack(
            T.tensor([reward]),
            T.tensor([game.episode]),
            T.tensor([done]),
            T.tensor([i_am_player])
        )
        self.reward_buffer.append(rew_pack)
        self.ticks_since_flush += 1
        if not done:
            self._flush_buffer(False, i_am_player, force=False)

