from abc import abstractmethod
from collections import deque
from random import randint
from typing import Tuple, Optional

import torch as T

from Environment import Game
from Learner.Nets import GameNet
from Learner.PrioReplayBuffer import PrioReplayBuffer
from Learner.Utils import get_cache_key, TensorDeque, Transition


class BaseAgent(PrioReplayBuffer):
    def __init__(self,
                 game: Game,
                 capacity: int,
                 alpha: float,
                 beta: float,
                 max_seq_length: int = 100,
                 burn_in_length: int = 20,
                 history_display: int = 100
                 ) -> None:
        super().__init__(capacity, max_seq_length, alpha, beta)
        self._last_transition: Transition | None = None

        self.game = game
        self.max_seq_length = max_seq_length
        self.burn_in_length = burn_in_length

        self.state_seq = TensorDeque(max_len=max_seq_length)
        self.action_seq = TensorDeque(max_len=max_seq_length)
        self.reward_seq = TensorDeque(max_len=max_seq_length, queue_like=T.zeros((1,), dtype=T.float))

        self.lstm_state_seq = TensorDeque(max_len=max_seq_length, queue_like=T.zeros((1, 1, 32), dtype=T.float))
        self.lstm_cell_seq = TensorDeque(max_len=max_seq_length, queue_like=T.zeros((1, 1, 32), dtype=T.float))

        self.has_won: bool = False
        self.episode_rewards = []
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

        self.lstm_state_seq.clear()
        self.lstm_cell_seq.clear()

        self.episode_rewards.clear()

    def flush_buffer(self, done: bool, i_am_player: int, force: bool = False) -> None:
        has_length = len(self.state_seq) >= (self.max_seq_length - self.burn_in_length)
        if has_length | force:
            self.add(
                self.state_seq,
                self.action_seq,
                self.reward_seq,
                self.lstm_state_seq,
                self.lstm_cell_seq,
                done,
                self.game.episode,
                i_am_player
            )

    def update_reward(self, reward: float, done: bool, i_am_player: int) -> None:
        self._to_buffer(reward=T.Tensor([reward]))
        self.episode_rewards.append(reward)
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
            lstm_state: Optional[T.Tensor] = None,
            lstm_cell: Optional[T.Tensor] = None
    ) -> None:
        if state is not None:
            self.state_seq.append(state)
        if action is not None:
            self.action_seq.append(action)
        if reward is not None:
            self.reward_seq.append(reward)
        if lstm_state is not None:
            self.lstm_state_seq.append(lstm_state)
        if lstm_cell is not None:
            self.lstm_cell_seq.append(lstm_cell)

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


class RandomAgent(BaseAgent):
    def __init__(self, game: Game, capacity: int, alpha: float, beta: float, history_display: int = 100) -> None:
        super().__init__(game, capacity, alpha, beta)

    def __repr__(self):
        return "RandomAgent"

    def sample_action(self, state: T.Tensor, i_am_player: int) -> tuple[T.Tensor, T.Tensor]:
        self._to_buffer(state=state)
        self._to_buffer(lstm_state=T.zeros((1, 1, 32), dtype=T.float))
        self._to_buffer(lstm_cell=T.zeros((1, 1, 32), dtype=T.float))

        if self.game.first_turn:
            if self.game.first_turn_village_switch:
                action_type = 2
            else:
                action_type = 1
        else:
            action_type = randint(0, 2)

        if action_type == 0:
            self._to_buffer(action=T.tensor((73, 73)))
            return T.tensor((0, 0)), T.tensor((73, 73))

        elif action_type == 1:
            road_mask = self.game.board.sparse_road_mask(
                i_am_player,
                self.game.players[i_am_player].hand,
                self.game.first_turn
            )
            if road_mask.numel() > 0:
                index, raw_index = self._sample_road(road_mask)
                self._to_buffer(action=raw_index)
                return T.tensor((1, index)), raw_index
            else:
                self._to_buffer(action=T.tensor((73, 73)))
                return T.tensor((0, 0)), T.tensor((73, 73))

        elif action_type == 2:
            village_mask = self.game.board.sparse_village_mask(
                i_am_player,
                self.game.players[i_am_player].hand,
                self.game.first_turn
            )
            if village_mask.numel() > 0:
                index = self._sample_village(village_mask)
                self._to_buffer(action=T.tensor((index, index)))
                return T.tensor((action_type, index)), T.tensor((index, index))
            else:
                self._to_buffer(action=T.tensor((73, 73)))
                return T.tensor((0, 0)), T.tensor((73, 73))
        else:
            raise RuntimeError(f"Random Agent chose illegal action {action_type}")

    # def update_reward(self, reward: float, done: bool, i_am_player: int):
    #     if reward < 0:
    #         breakpoint()
    #     super().update_reward(reward, done, i_am_player)

    def clear_cache(self):
        pass

    @staticmethod
    def _sample_village(village_mask) -> int:
        if village_mask.numel() > 0:
            # available_villages = village_mask.argwhere().squeeze()
            if village_mask.numel() == 1:
                return village_mask.item()
            else:
                return village_mask[T.randint(0, village_mask.shape[0], (1,))]
        else:
            raise RuntimeError("Random Agent could not find house on round 1")

    def _sample_road(self, road_mask: T.Tensor) -> Tuple[int, T.Tensor]:
        if road_mask.numel() > 0:
            # available_roads = road_mask.argwhere().squeeze()
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


class QAgent(BaseAgent):
    def __init__(self, q_net: GameNet, game: Game, capacity: int, alpha: float, beta: float,
                 history_display: int = 100) -> None:
        super().__init__(game, capacity, alpha, beta, history_display)

        self.q_net = q_net
        self.sparse_edge = game.board.state.edge_index.clone()
        self.empty_edge = T.zeros((self.sparse_edge.shape[1],), dtype=T.bool)
        self.empty_node = T.zeros((54,), dtype=T.bool)

        self.action_mask = q_net.action_mask.clone()

        self.action = T.tensor((0, 0))
        self.raw_action = T.tensor((73, 73))

        self.episode_state_action_cache = {}

    def __repr__(self) -> str:
        return "QAgent"

    def sample_action(self, state: T.Tensor, i_am_player: int) -> Tuple[T.Tensor, T.Tensor]:
        build_q: T.Tensor

        state_key = get_cache_key(state)

        self._to_buffer(state=state)

        if self.lstm_state_seq.is_empty:
            self._to_buffer(lstm_state=T.zeros((1, 1, 32), dtype=T.float))
            self._to_buffer(lstm_cell=T.zeros((1, 1, 32), dtype=T.float))

        h0 = self.lstm_state_seq[-1]
        c0 = self.lstm_cell_seq[-1]

        with T.no_grad():
            q, hn, cn = self.q_net(state.unsqueeze(1), T.Tensor([1]), h0, c0)
            self._to_buffer(lstm_state=hn)
            self._to_buffer(lstm_cell=cn)

        pass_q = q[0, 0, -1, -1, i_am_player]
        q = q[0, 0, :54, :54, i_am_player]
        q = self._pull_cache(state_key, q)

        build_q = q.max()

        if (pass_q > build_q) & (not self.game.first_turn):
            self._to_buffer(action=T.tensor((73, 73)))
            return T.tensor((0, 0)), T.tensor((73, 73))
        else:
            build_action = T.argwhere(q == build_q).cpu()
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
                self._to_buffer(action=build_action)
                self._push_cache(state_key, build_action)
                return T.tensor((1, index)), build_action

            elif build_action[0] == build_action[1]:
                if build_action[0] >= 54:
                    raise RuntimeError("Non-node index returned for building settlement")
                self._to_buffer(action=build_action)
                self._push_cache(state_key, build_action)
                return T.tensor((2, build_action[0])), build_action
            else:
                raise RuntimeError("Invalid Build Action in QAgent")

    def clear_cache(self) -> None:
        self.episode_state_action_cache = {}

    def _push_cache(self, state_key, build_action):
        if state_key not in self.episode_state_action_cache:
            self.episode_state_action_cache[state_key] = [build_action]
        else:
            self.episode_state_action_cache[state_key].append(build_action)

    def _pull_cache(self, state_key: Tuple, q: T.Tensor) -> T.Tensor:
        if state_key in self.episode_state_action_cache:
            cache_acts = self.episode_state_action_cache[state_key]
            for act in cache_acts:
                q[act[0], act[1]] = -T.inf
        return q
