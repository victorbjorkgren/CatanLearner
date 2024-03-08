from abc import abstractmethod
from collections import deque
from random import randint, random
from typing import Tuple, Optional

import torch as T

from Environment import Game
from Learner.Loss import Loss
from Learner.Nets import GameNet
from Learner.PrioReplayBuffer import PrioReplayBuffer
from Learner.Utils import TensorDeque, Transition, TensorUtils
from Learner.constants import USE_ACTOR_PRIO, GAMMA


class Agent:
    def __init__(self,
                 game: Game,
                 buffer: PrioReplayBuffer,
                 max_seq_length: int,
                 burn_in_length: int,
                 history_display: int = 100
                 ) -> None:
        # super().__init__(capacity, max_seq_length, alpha, beta)
        self._last_transition: Transition | None = None
        self.tracker_instance = None

        self.game = game
        self.buffer = buffer
        self.max_seq_length = max_seq_length
        self.burn_in_length = burn_in_length

        self.state_seq = TensorDeque(max_len=max_seq_length)
        self.q_seq = TensorDeque(max_len=max_seq_length)
        self.action_seq = TensorDeque(max_len=max_seq_length)
        self.reward_seq = TensorDeque(max_len=max_seq_length, queue_like=T.zeros((1,), dtype=T.float))
        self.was_trade_seq = TensorDeque(max_len=max_seq_length)

        self.lstm_state_seq = TensorDeque(max_len=max_seq_length, queue_like=T.zeros((1, 1, 32), dtype=T.float))
        self.lstm_cell_seq = TensorDeque(max_len=max_seq_length, queue_like=T.zeros((1, 1, 32), dtype=T.float))

        self.has_won: bool = False
        self.episode_score: int = 0
        self.house_history = deque(maxlen=history_display)
        self.win_history = deque(maxlen=history_display)
        self.beat_time = deque(maxlen=history_display)

    def __repr__(self):
        return "BaseAgent"

    @abstractmethod
    def sample_action(self, state: T.Tensor, i_am_player: int) -> tuple[T.Tensor, T.Tensor]:
        pass

    @abstractmethod
    def clear_cache(self):
        pass

    def sample_random(self, i_am_player: int) -> Tuple[T.Tensor, T.Tensor, bool]:
        if self.game.first_turn:
            if self.game.first_turn_village_switch:
                action_type = 2
            else:
                action_type = 1
        else:
            action_type = randint(0, 3)

        if action_type == 0:
            return T.tensor((0, 0)), T.tensor((73, 73)), False

        elif action_type == 1:
            road_mask = self.game.board.sparse_road_mask(
                i_am_player,
                self.game.players[i_am_player].hand,
                self.game.first_turn
            )
            if road_mask.numel() > 0:
                index, raw_index = self._sample_road(road_mask)
                return T.tensor((1, index)), raw_index, False
            else:
                return T.tensor((0, 0)), T.tensor((73, 73)), False

        elif action_type == 2:
            village_mask = self.game.board.sparse_village_mask(
                i_am_player,
                self.game.players[i_am_player].hand,
                self.game.first_turn
            )
            if village_mask.numel() > 0:
                index = self._sample_village(village_mask)
                return T.tensor((action_type, index)), T.tensor((index, index)), False
            else:
                return T.tensor((0, 0)), T.tensor((73, 73)), False

        elif action_type == 3:
            tradeable = self.game.can_trade(i_am_player, 4)
            desired = randint(0, 4)
            if tradeable.numel() == 0:
                return T.tensor((0, 0)), T.tensor((73, 73)), False
            elif tradeable.numel() == 1:
                return T.Tensor((3, tradeable[0], desired)), T.Tensor((tradeable[0], desired)), True
            else:
                given = randint(0, tradeable.shape[0] - 1)
                return T.Tensor((3, tradeable[given], desired)), T.Tensor((tradeable[given], desired)), True
        else:
            raise RuntimeError(f"Illegal action type chosen {action_type}")

    @staticmethod
    def _sample_village(village_mask) -> int:
        if village_mask.numel() > 0:
            if village_mask.numel() == 1:
                return village_mask.item()
            else:
                return village_mask[T.randint(0, village_mask.shape[0], (1,))]
        else:
            raise RuntimeError("Random Agent could not find house on round 1")

    def _sample_road(self, road_mask: T.Tensor) -> Tuple[int, T.Tensor]:
        if road_mask.numel() > 0:
            if road_mask.numel() == 2:
                index = road_mask
            else:
                index = road_mask[:, T.randint(0, road_mask.shape[1], (1,))]
            edge_index = (
                    (self.game.board.state.edge_index[0, :] == index[0, :])
                    & (self.game.board.state.edge_index[1, :] == index[1, :])
            ).nonzero()
            return edge_index.item(), index.squeeze()
        else:
            raise RuntimeError("Random Agent could not find road on round 1")

    def signal_episode_done(self, i_am_player: int) -> None:
        self.flush_buffer(True, i_am_player, force=True)

        if self.has_won:
            self.win_history.append(1)
            self.beat_time.append(self.game.turn)
        else:
            self.win_history.append(0)
        self.has_won = False

        self.state_seq.clear()
        self.action_seq.clear()
        self.reward_seq.clear()
        self.was_trade_seq.clear()
        self.q_seq.clear()

        self.lstm_state_seq.clear()
        self.lstm_cell_seq.clear()

        self.episode_score = 0

    def flush_buffer(self, done: bool, i_am_player: int, force: bool = False) -> None:
        has_length = len(self.state_seq) >= (self.max_seq_length - self.burn_in_length)
        if has_length | force:
            state = self.state_seq.to_tensor()
            action = self.action_seq.to_tensor()
            reward = self.reward_seq.to_tensor()
            was_trade = self.was_trade_seq.to_tensor()
            lstm_state = self.lstm_state_seq.to_tensor()
            lstm_cell = self.lstm_cell_seq.to_tensor()
            q = self.q_seq.to_tensor()

            state = state.permute(1, 0, 2, 3, 4)
            action = action.long()
            reward = reward.permute(1, 0)
            q = q.unsqueeze(0)
            lstm_state = lstm_state[:-1, 0, 0, :]
            lstm_cell = lstm_cell[:-1, 0, 0, :]

            if USE_ACTOR_PRIO:
                # raise NotImplementedError("Using prio actor is stale. Trade has not been implemented here")
                lstm_target_state = lstm_state[None, None, -1, :]
                lstm_target_cell = lstm_cell[None, None, -1, :]

                seq_len = T.tensor([state.shape[1]], dtype=T.long)
                titan_q_net = self.tracker_instance.get_titan().q_net

                td_error = Loss.get_td_error(
                    titan_q_net,
                    q,
                    state,
                    reward,
                    GAMMA,
                    None,
                    seq_len,
                    lstm_target_state,
                    lstm_target_cell,
                    T.tensor([done], dtype=T.bool),
                    T.tensor([i_am_player], dtype=T.long)
                )
            else:
                td_error = T.tensor([1.], dtype=T.float)

            self.buffer.add(
                state,
                action,
                was_trade,
                reward,
                td_error.mean(),
                lstm_state,
                lstm_cell,
                done,
                self.game.episode,
                i_am_player
            )

    def update_reward(self, reward: float, done: bool, i_am_player: int) -> None:
        self._to_buffer(reward=T.Tensor([reward]))
        if reward > 0.:
            self.episode_score += 1
        if not done:
            self.flush_buffer(False, i_am_player, force=False)

    def register_victory(self) -> None:
        self.has_won = True

    def _to_buffer(
            self,
            *,
            state: Optional[T.Tensor] = None,
            action: Optional[T.Tensor] = None,
            reward: Optional[T.Tensor] = None,
            q: Optional[T.Tensor] = None,
            lstm_state: Optional[T.Tensor] = None,
            lstm_cell: Optional[T.Tensor] = None,
            was_trade: Optional[T.Tensor] = None
    ) -> None:
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

    @property
    def avg_reward(self) -> float:
        if len(self.reward_seq) > 0:
            reward_history = self.reward_seq.to_tensor()
            # n = (reward_history > 0).sum().item()
            # n = max(1, n)
            return reward_history.sum().item()  # / n
        else:
            return 0.

    @property
    def avg_beat_time(self) -> float:
        if len(self.beat_time) > 0:
            return sum(self.beat_time) / len(self.beat_time)
        else:
            return 0.

    @property
    def sum_win(self):
        return sum(self.win_history)


class RandomAgent:
    def __init__(self) -> None:
        raise AssertionError("RandomAgent is stale!")

    def __repr__(self):
        return "RandomAgent"

    def sample_action(self, state: T.Tensor, i_am_player: int) -> tuple[T.Tensor, T.Tensor]:
        # self._to_buffer(state=state)
        # self._to_buffer(lstm_state=T.zeros((1, 1, 32), dtype=T.float))
        # self._to_buffer(lstm_cell=T.zeros((1, 1, 32), dtype=T.float))

        # return self.sample_random(i_am_player)
        pass

    def clear_cache(self):
        pass

    @staticmethod
    def _sample_village(village_mask) -> int:
        if village_mask.numel() > 0:
            if village_mask.numel() == 1:
                return village_mask.item()
            else:
                return village_mask[T.randint(0, village_mask.shape[0], (1,))]
        else:
            raise RuntimeError("Random Agent could not find house on round 1")

    def _sample_road(self, road_mask: T.Tensor) -> Tuple[int, T.Tensor]:
        # if road_mask.numel() > 0:
        #     # available_roads = road_mask.argwhere().squeeze()
        #     if road_mask.numel() == 2:
        #         index = road_mask
        #     else:
        #         index = road_mask[:, T.randint(0, road_mask.shape[1], (1,))]
        #     edge_index = (
        #             (self.game.board.state.edge_index[0, :] == index[0, :])
        #             & (self.game.board.state.edge_index[1, :] == index[1, :])
        #     ).nonzero()
        #     return edge_index.item(), index.squeeze()
        # else:
        #     raise RuntimeError("Random Agent could not find road on round 1")
        pass


class QAgent(Agent):
    def __init__(
            self,
            q_net: GameNet,
            game: Game,
            buffer: PrioReplayBuffer,
            max_sequence_length: int,
            burn_in_length: int,
            is_titan: bool = False,
            history_display: int = 100
    ) -> None:

        super().__init__(game, buffer, max_sequence_length, burn_in_length, history_display)

        self.q_net = q_net
        self.is_titan = is_titan
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

    def __repr__(self) -> str:
        return f"QAgent_{self.my_suffix}"

    def set_titan(self):
        self.is_titan = True

    def unset_titan(self):
        self.is_titan = False

    def load_state(self, suffix, epsilon, elo):
        if self.is_titan:
            self.q_net.load('latest')
            self.epsilon = 1.
            self.my_suffix = f'Titan ({int(elo)})'
            self.my_name = 'Titan'
        elif (epsilon == 0.) | (suffix == 'Random'):
            self.my_name = 'Random'
            self.my_suffix = f'Random ({int(elo)})'
            self.epsilon = 0.
        else:
            self.q_net.load(suffix)
            self.my_name = suffix
            self.epsilon = epsilon
            str_start = len("Q_Agent") + 1
            str_end = len(".pth")
            self.my_suffix = f"{suffix[str_start: -str_end]}_{epsilon:.2f} ({int(elo)})"

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

        with T.no_grad():
            q_mat, q_trade_mat, hn, cn = self.q_net(state.unsqueeze(1), T.Tensor([1]), h0, c0)
            self._to_buffer(lstm_state=hn)
            self._to_buffer(lstm_cell=cn)

        if random() > self.epsilon:
            action, raw_action, was_trade = self.sample_random(i_am_player)
            if was_trade:
                self._to_buffer(action=T.Tensor((action[1], action[2])))
            else:
                self._to_buffer(action=raw_action)
                self._to_buffer(q=q_mat[0, 0, raw_action[0], raw_action[1], i_am_player])
            self._to_buffer(was_trade=T.tensor([was_trade], dtype=T.bool))
            return action, raw_action

        pass_q = q_mat[0, 0, -1, -1, i_am_player]
        q_trade_mat = q_trade_mat[0, 0, :, :, i_am_player]
        q_mat = q_mat[0, 0, :54, :54, i_am_player]
        q_mat, q_trade_mat = self._pull_cache(state_key, q_mat, q_trade_mat)

        build_q = q_mat.max()
        trade_q = q_trade_mat[0].max() + q_trade_mat[1].max()

        if (pass_q > build_q) & (pass_q > trade_q) & (not self.game.first_turn):
            self._to_buffer(q=pass_q)
            self._to_buffer(action=T.tensor((73, 73)))
            self._to_buffer(was_trade=T.tensor([False], dtype=T.bool))
            return T.tensor((0, 0)), T.tensor((73, 73))
        elif (trade_q > build_q) & (not self.game.first_turn):
            give = T.argwhere(q_trade_mat[0] == q_trade_mat[0].max()).cpu()
            get = T.argwhere(q_trade_mat[1] == q_trade_mat[1].max()).cpu()
            trade_action = T.tensor((give, get), dtype=T.float)
            self._to_buffer(q=trade_q)
            self._to_buffer(action=trade_action)
            self._to_buffer(was_trade=T.tensor([True], dtype=T.bool))
            self._push_cache(state_key, trade_action, True)
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
                return T.tensor((1, index)), build_action

            elif build_action[0] == build_action[1]:
                if build_action[0] >= 54:
                    raise RuntimeError("Non-node index returned for building settlement")
                self._to_buffer(q=build_q)
                self._to_buffer(action=build_action)
                self._to_buffer(was_trade=T.tensor([False], dtype=T.bool))
                self._push_cache(state_key, build_action, False)
                return T.tensor((2, build_action[0])), build_action
            else:
                raise RuntimeError("Invalid Build Action in QAgent")

    def clear_cache(self) -> None:
        self.episode_state_action_cache = {}

    def _push_cache(self, state_key, action, was_trade):
        if state_key not in self.episode_state_action_cache:
            self.episode_state_action_cache[state_key] = [(was_trade, action)]
        else:
            self.episode_state_action_cache[state_key].append((was_trade, action))

    def _pull_cache(self, state_key: Tuple, q: T.Tensor, trade_q: T.Tensor) -> T.Tensor:
        if state_key in self.episode_state_action_cache:
            cache_acts = self.episode_state_action_cache[state_key]
            for was_trade, act in cache_acts:
                if was_trade:
                    trade_q[0, act[0].long()] = -T.inf
                    trade_q[1, act[1].long()] = -T.inf
                else:
                    q[act[0], act[1]] = -T.inf
        return q, trade_q
