from collections import deque
from random import randint
from typing import Tuple

import torch as T
import torch_geometric as pyg

from Learner.Nets import GameNet
from Environment import Game
from Learner.PrioReplayBuffer import PrioReplayBuffer

from Learner.Utils import get_masks, get_cache_key


class BaseAgent(PrioReplayBuffer):
    def __init__(self, game: Game, capacity: int, alpha: float, beta: float, history_display: int = 100) -> None:
        super().__init__(capacity, alpha, beta)
        self.game = game

        self.reward_history = deque(maxlen=history_display)
        self.house_history = deque(maxlen=history_display)
        self.win_history = deque(maxlen=history_display)
        self.beat_time = deque(maxlen=history_display)

    def sample_action(self, game: Game, state: T.Tensor, i_am_player: int, remember: bool) -> tuple[T.Tensor, T.Tensor]:
        pass

    def sample_village(self, game: Game, village_mask: T.Tensor, i_am_player: int):
        pass

    def sample_road(self, game: Game, road_mask: T.Tensor, i_am_player: int):
        pass

    def clear_cache(self) -> None:
        pass

    def update_reward(self, reward: float, done: bool) -> None:
        super().update_reward(reward, done)
        self.reward_history.append(reward)
        if done & (reward > 0.):
            self.win_history.append(1)
            self.beat_time.append(self.game.turn)

    @property
    def avg_reward(self) -> float:
        if len(self.reward_history) > 0:
            return sum(self.reward_history) / len(self.reward_history)
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

    def __repr__(self):
        return "BaseAgent"


class RandomAgent(BaseAgent):
    def __init__(self, game: Game, capacity: int, alpha: float, beta: float, history_display: int = 100) -> None:
        super().__init__(game, capacity, alpha, beta)

    def sample_action(self, game: Game, state: T.Tensor, i_am_player: int, remember: bool = True) -> tuple[
        T.Tensor, T.Tensor]:
        road_mask, village_mask = get_masks(game, i_am_player)

        if game.first_turn:
            if game.first_turn_village_switch:
                index = self.sample_village(game, village_mask, i_am_player)
                if index is None:
                    raise IndexError("No index returned from sampled village")
                return T.tensor((2, index)), T.tensor((index, index))
            else:
                index = self.sample_road(game, road_mask, i_am_player)
                return T.tensor((1, index)), game.board.state.edge_index[:, index]

        action_type = randint(0, 2)

        if action_type == 0:
            return T.tensor((0, 0)), T.tensor((73, 73))

        elif action_type == 1:
            if road_mask.sum() > 0:
                available_roads = road_mask.argwhere().squeeze()
                if road_mask.sum() == 1:
                    index = available_roads.item()
                else:
                    index = available_roads[T.randint(0, len(available_roads), (1,))]
                return T.tensor((1, index)), game.board.state.edge_index[:, index]
            else:
                return T.tensor((0, 0)), T.tensor((73, 73))

        elif action_type == 2:
            if village_mask.sum() > 0:
                available_villages = village_mask.argwhere().squeeze()
                if village_mask.sum() == 1:
                    index = available_villages.item()
                else:
                    index = available_villages[T.randint(0, len(available_villages), (1,))]
                return T.tensor((action_type, index)), T.tensor((index, index))
            else:
                return T.tensor((0, 0)), T.tensor((73, 73))
        else:
            raise Exception("Random Agent chose illegal action")

    def sample_village(self, game, village_mask, i_am_player) -> int:
        if village_mask.sum() > 0:
            available_villages = village_mask.argwhere().squeeze()
            if village_mask.sum() == 1:
                index = available_villages.item()
            else:
                index = available_villages[T.randint(0, len(available_villages), (1,))]
            return index
        else:
            raise Exception("Random Agent could not find house on round 1")

    def sample_road(self, game, road_mask, i_am_player) -> int:
        if road_mask.sum() > 0:
            available_roads = road_mask.argwhere().squeeze()
            if road_mask.sum() == 1:
                index = available_roads.item()
            else:
                index = available_roads[T.randint(0, len(available_roads), (1,))]
            return index
        else:
            raise Exception("Random Agent could not find road on round 1")

    def __repr__(self):
        return "RandomAgent"


class QAgent(BaseAgent):
    def __init__(self, q_net: GameNet, game: Game, capacity: int, alpha: float, beta: float, history_display: int = 100) -> None:
        super().__init__(game, capacity, alpha, beta, history_display)
        self.q_net = q_net
        self.sparse_edge = game.board.state.edge_index.clone()
        self.empty_edge = T.zeros((self.sparse_edge.shape[1],), dtype=T.bool)
        self.empty_node = T.zeros((54,), dtype=T.bool)

        self.action_mask = q_net.action_mask.clone()

        self.action = T.tensor((0, 0))
        self.raw_action = T.tensor((73, 73))

        self.episode_state_action_cache = {}

    def sample_action(self, game: Game, state: T.Tensor, i_am_player: int, remember: bool = True) -> tuple[
        T.Tensor, T.Tensor]:
        build_q: T.Tensor
        state_key = get_cache_key(state)

        with T.no_grad():
            q = self.q_net(state).detach()
        pass_q = q[0, -1, -1, i_am_player]
        q = q[0, :54, :54, i_am_player]
        q = self.pull_cache(state_key, q)

        build_q = q.max()

        if (pass_q > build_q) & (not game.first_turn):
            if remember:
                self.add(state, self.raw_action, 0, 0, game.episode, i_am_player)
            return T.tensor((0, 0)), T.tensor((73, 73))
        else:
            build_action = T.argwhere(q == build_q).cpu()
            if build_action.shape[0] > 1:
                build_action = build_action[T.randint(0, build_action.shape[0], (1,))].squeeze()
            elif build_action.shape[0] == 1:
                build_action = build_action.squeeze()
            else:
                raise Exception("Invalid Build Action in QAgent")

            if build_action[0] != build_action[1]:
                bool_hit = (
                        (game.board.state.edge_index[0] == build_action[0])
                        & (game.board.state.edge_index[1] == build_action[1])
                )
                index = T.argwhere(bool_hit).item()
                if remember:
                    self.add(state, build_action, 0, 0, game.episode, i_am_player)
                self.push_cache(state_key, build_action)
                return T.tensor((1, index)), build_action

            elif build_action[0] == build_action[1]:
                if build_action[0] >= 54:
                    raise Exception("Non-node index returned for building settlement")
                if remember:
                    self.add(state, build_action, 0, 0, game.episode, i_am_player)
                self.push_cache(state_key, build_action)
                return T.tensor((2, build_action[0])), build_action
            else:
                raise Exception("Invalid Build Action in QAgent")

    def mask_to_dense(self, road_mask: T.Tensor, village_mask: T.Tensor) -> T.Tensor:
        village_mask = T.diag_embed(village_mask).bool()
        if road_mask.sum() == 0:
            road_mask = T.zeros_like(village_mask, dtype=T.bool)
        else:
            road_mask = pyg.utils.to_dense_adj(self.sparse_edge[:, road_mask],
                                               max_num_nodes=village_mask.shape[-1]).bool()
        return village_mask + road_mask

    def clear_cache(self) -> None:
        self.episode_state_action_cache = {}

    def __repr__(self) -> str:
        return "QAgent"

    def push_cache(self, state_key, build_action):
        if state_key not in self.episode_state_action_cache:
            self.episode_state_action_cache[state_key] = [build_action]
        else:
            self.episode_state_action_cache[state_key].append(build_action)

    def pull_cache(self, state_key: Tuple, q: T.Tensor) -> T.Tensor:
        if state_key in self.episode_state_action_cache:
            cache_acts = self.episode_state_action_cache[state_key]
            for act in cache_acts:
                q[act[0], act[1]] = -T.inf
        return q
