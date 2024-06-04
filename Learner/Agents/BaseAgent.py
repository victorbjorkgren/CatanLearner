from abc import abstractmethod
from collections import deque
from random import randint
from typing import Tuple, Optional

import torch as T

from Environment import Game
from Learner.Loss import Loss
from Learner.PrioReplayBuffer import PrioReplayBuffer
from Learner.Utility.Utils import TensorDeque, Transition
from Learner.constants import USE_ACTOR_PRIO, GAMMA


class BaseAgent:
    def __init__(self,
                 game: Game,
                 buffer: PrioReplayBuffer,
                 max_seq_length: int,
                 burn_in_length: int,
                 is_titan: False,
                 history_display: int = 100
                 ) -> None:
        self.name = 'BaseAgent'
        self.my_suffix = '__'
        self.n_trades = 0
        self.n_roads = 0
        self.n_houses = 0
        self._last_transition: Transition | None = None
        self.tracker_instance = None

        self.is_titan = is_titan
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
        self.fail_count = 0
        self.win_history = deque(maxlen=history_display)
        self.score_history = deque(maxlen=history_display)
        self.beat_time = deque(maxlen=history_display)

    def __repr__(self) -> str:
        return f"{self.name}_{self.my_suffix}"

    @abstractmethod
    def sample_action(self, state: T.Tensor, i_am_player: int) -> tuple[T.Tensor, T.Tensor]:
        pass

    @abstractmethod
    def clear_cache(self):
        pass

    @abstractmethod
    def update_reward(self, reward: float, done: bool, game, i_am_player: int) -> None:
        raise NotImplementedError

    def load_state(self, suffix, elo):
        if self.is_titan:
            self.net.load('latest')
            # self.epsilon = 1.
            self.my_suffix = f'Titan ({int(elo)})'
            self.my_name = 'latest'
        # elif (epsilon == 0.) | (suffix == 'Random'):
        #     self.my_name = 'Random'
        #     self.my_suffix = f'Random ({int(elo)})'
        #     self.epsilon = 0.
        else:
            self.net.load(suffix)
            self.my_name = suffix
            # self.epsilon = epsilon
            str_start = len(self.name) + 1
            str_end = len(".pth")
            # self.my_suffix = f"{suffix[str_start: -str_end]}_{epsilon:.2f} ({int(elo)})"
            self.my_suffix = f"{suffix[str_start: -str_end]} ({int(elo)})"

    def register_action(self, action: int) -> None:
        if action == 0:
            pass
        elif action == 1:
            self.n_houses += 1
        elif action == 2:
            self.n_roads += 1
        elif action == 3:
            self.n_trades += 1
        else:
            raise RuntimeError("Tried to register faulty action")

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
            raise RuntimeError("Random BaseAgent could not find road on round 1")

    def signal_episode_done(self, i_am_player: int) -> None:
        # self._flush_buffer(True, i_am_player, force=True)

        if self.has_won:
            self.win_history.append(1)
            self.beat_time.append(self.game.turn)
        else:
            self.win_history.append(0)
        self.score_history.append(self.game.players[i_am_player].points)
        self.has_won = False

        # self.state_seq.clear()
        # self.action_seq.clear()
        # self.reward_seq.clear()
        # self.was_trade_seq.clear()
        # self.q_seq.clear()

        # self.lstm_state_seq.clear()
        # self.lstm_cell_seq.clear()

        self.episode_score = 0
        self.n_houses = 0
        self.n_roads = 0
        self.n_trades = 0

    def _flush_buffer(self, done: bool, i_am_player: int, force: bool = False) -> None:
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

    def register_victory(self) -> None:
        self.has_won = True

    def signal_failure(self, failed):
        if failed & (not self.game.first_turn):
            self.fail_count += 1
        else:
            self.fail_count = 0

    def set_titan(self):
        self.is_titan = True

    def unset_titan(self):
        self.is_titan = False

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

    @staticmethod
    def _sample_village(village_mask) -> int:
        if village_mask.numel() > 0:
            if village_mask.numel() == 1:
                return village_mask.item()
            else:
                return village_mask[T.randint(0, village_mask.shape[0], (1,))]
        else:
            raise RuntimeError("Random BaseAgent could not find house on round 1")

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
    def avg_score(self) -> float:
        return round(sum(self.score_history) / max(len(self.score_history), 1), 1)

    @property
    def avg_beat_time(self) -> float:
        if len(self.beat_time) > 0:
            return sum(self.beat_time) / len(self.beat_time)
        else:
            return 0.

    @property
    def sum_win(self):
        return sum(self.win_history)

    @property
    def episode_actions(self) -> T.Tensor:
        return T.tensor([self.n_houses, self.n_roads, self.n_trades])
