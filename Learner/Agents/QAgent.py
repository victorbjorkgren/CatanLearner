from random import random
from typing import Tuple, Optional

import torch as T

from Environment import Game
from Learner.Agents.BaseAgent import BaseAgent
from Learner.Nets import GameNet
from Learner.PrioReplayBuffer import PrioReplayBuffer
from Learner.Utility.Utils import TensorUtils
from Learner.constants import N_PLAYERS, FAILURE_ALLOWANCE


class QAgent(BaseAgent):
    def __init__(self,
                 q_net: GameNet,
                 game: Game,
                 buffer: PrioReplayBuffer,
                 max_sequence_length: int,
                 burn_in_length: int,
                 is_titan: bool = False,
                 history_display: int = 100) -> None:
        super().__init__(game, buffer, max_sequence_length, burn_in_length, is_titan, history_display)
        self.name = 'Q_Agent'
        self.q_net = q_net
        self.epsilon = 1.

        self.my_suffix = None
        self.my_name = None

        self.sparse_edge = game.board.state.edge_index.clone()
        self.empty_edge = T.zeros((self.sparse_edge.shape[1],), dtype=T.bool)
        self.empty_node = T.zeros((54,), dtype=T.bool)

        self.action_mask = q_net.action_mask.clone()

        self.action = T.tensor((0, 0))
        self.raw_action = T.tensor((73, 73))

        self.episode_state_action_cache = {}

    def sample_action(self, state: T.Tensor, i_am_player: int) -> Tuple[T.Tensor, T.Tensor]:
        build_q: T.Tensor
        q_trade_mat: T.Tensor

        state_key = TensorUtils.get_cache_key(state)

        self._to_buffer(state=state)

        if self.lstm_state_seq.is_empty:
            self._to_buffer(lstm_state=T.zeros((1, 1, 32), dtype=T.float))
            self._to_buffer(lstm_cell=T.zeros((1, 1, 32), dtype=T.float))

        h0 = self.lstm_state_seq[-1]
        c0 = self.lstm_cell_seq[-1]

        # Get masks
        build_mask = ~state[:, :, :, -N_PLAYERS + i_am_player].bool()
        tradeable = self.game.can_trade(i_am_player, 4)
        trade_mask = ~T.isin(T.arange(5), tradeable)
        with T.no_grad():
            q_mat, q_trade_mat, hn, cn = self.q_net(state.unsqueeze(1), T.Tensor([1]), h0, c0)
            self._to_buffer(lstm_state=hn)
            self._to_buffer(lstm_cell=cn)

        if random() > self.epsilon:
            action, raw_action, was_trade = self.sample_random(i_am_player)
            if was_trade:
                give_q = q_trade_mat[0, 0, 0, raw_action[0].long(), i_am_player]
                get_q = q_trade_mat[0, 0, 1, raw_action[1].long(), i_am_player]
                self._to_buffer(action=T.Tensor((action[1], action[2])))
                self._to_buffer(q=give_q + get_q)
            else:
                self._to_buffer(action=raw_action)
                self._to_buffer(q=q_mat[0, 0, raw_action[0], raw_action[1], i_am_player])
            self._to_buffer(was_trade=T.tensor([was_trade], dtype=T.bool))
            self.register_action(action[0])
            return action, raw_action

        pass_q = q_mat[0, 0, -1, -1, i_am_player]

        q_mat[0, build_mask] = -T.inf
        q_trade_mat[0, 0, 0, :, i_am_player][trade_mask] = -T.inf

        q_trade_mat = q_trade_mat[0, 0, :, :, i_am_player]
        q_mat = q_mat[0, 0, :54, :54, i_am_player]
        q_mat, q_trade_mat = self._pull_cache(state_key, q_mat, q_trade_mat)

        build_q = q_mat.max()
        trade_q = q_trade_mat[0].max() + q_trade_mat[1].max()

        pass_q_is_max = (pass_q > build_q) & (pass_q > trade_q)
        fail_allowance_reached = (self.fail_count > FAILURE_ALLOWANCE)
        not_first_turn = (not self.game.first_turn)

        if (pass_q_is_max | fail_allowance_reached) & not_first_turn:
            self._to_buffer(q=pass_q)
            self._to_buffer(action=T.tensor((73, 73)))
            self._to_buffer(was_trade=T.tensor([False], dtype=T.bool))
            return T.tensor((0, 0)), T.tensor((73, 73))
        elif (trade_q > build_q) & not_first_turn:
            give = T.argwhere(q_trade_mat[0] == q_trade_mat[0].max()).cpu()
            get = T.argwhere(q_trade_mat[1] == q_trade_mat[1].max()).cpu()
            try:
                trade_action = T.tensor((give, get), dtype=T.float)
            except:
                print("Exception in trade_action = T.tensor((give, get), dtype=T.float)")
            self._to_buffer(q=trade_q)
            self._to_buffer(action=trade_action)
            self._to_buffer(was_trade=T.tensor([True], dtype=T.bool))
            self._push_cache(state_key, trade_action, True)
            self.register_action(3)
            return T.Tensor((3, give, get)), trade_action
        else:
            build_action = T.argwhere(q_mat == build_q).cpu()
            if build_action.shape[0] > 1:
                build_action = build_action[T.randint(0, build_action.shape[0], (1,))].squeeze()
            elif build_action.shape[0] == 1:
                build_action = build_action.squeeze()
            else:
                raise RuntimeError("Invalid Build Action in QAgent")

            if build_action[0] != build_action[1]:
                bool_hit = (
                        (self.game.board.state.edge_index[0] == build_action[0])
                        & (self.game.board.state.edge_index[1] == build_action[1])
                )
                index = T.argwhere(bool_hit).item()
                self._to_buffer(q=build_q)
                self._to_buffer(action=build_action)
                self._to_buffer(was_trade=T.tensor([False], dtype=T.bool))
                self._push_cache(state_key, build_action, False)
                self.register_action(1)
                return T.tensor((1, index)), build_action

            elif build_action[0] == build_action[1]:
                if build_action[0] >= 54:
                    raise RuntimeError("Non-node index returned for building settlement")
                self._to_buffer(q=build_q)
                self._to_buffer(action=build_action)
                self._to_buffer(was_trade=T.tensor([False], dtype=T.bool))
                self._push_cache(state_key, build_action, False)
                self.register_action(2)
                return T.tensor((2, build_action[0])), build_action
            else:
                raise RuntimeError("Invalid Build Action in QAgent")

    def clear_cache(self) -> None:
        self.episode_state_action_cache = {}

    def _to_buffer(self,
                   *,
                   state: Optional[T.Tensor] = None,
                   action: Optional[T.Tensor] = None,
                   reward: Optional[T.Tensor] = None,
                   q: Optional[T.Tensor] = None,
                   lstm_state: Optional[T.Tensor] = None,
                   lstm_cell: Optional[T.Tensor] = None,
                   was_trade: Optional[T.Tensor] = None) -> None:
        if state is not None:
            self.state_seq.append(state)
        if action is not None:
            self.action_seq.append(action)
        if reward is not None:
            self.reward_seq.append(reward)
        if q is not None:
            self.q_seq.append(q)
        if lstm_state is not None:
            self.lstm_state_seq.append(lstm_state)
        if lstm_cell is not None:
            self.lstm_cell_seq.append(lstm_cell)
        if was_trade is not None:
            self.was_trade_seq.append(was_trade)

    def update_reward(self, reward: float, done: bool, i_am_player: int) -> None:
        self._to_buffer(reward=T.Tensor([reward]))
        if reward > 0.:
            self.episode_score += 1
        if not done:
            self._flush_buffer(False, i_am_player, force=False)

    def _push_cache(self, state_key, action, was_trade):
        if state_key not in self.episode_state_action_cache:
            self.episode_state_action_cache[state_key] = [(was_trade, action)]
        else:
            self.episode_state_action_cache[state_key].append((was_trade, action))

    def _pull_cache(self, state_key: Tuple, q: T.Tensor, trade_q: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        if state_key in self.episode_state_action_cache:
            cache_acts = self.episode_state_action_cache[state_key]
            for was_trade, act in cache_acts:
                if was_trade:
                    # Only make -inf for give resource. Errors are only in that aspect of the trade.
                    trade_q[0, act[0].long()] = -T.inf
                else:
                    q[act[0], act[1]] = -T.inf
        return q, trade_q
