from random import randint
import torch as T
import torch_geometric as pyg

from Learner.Nets import GameNet
from Environment import Game
from Learner.PrioReplayBuffer import PrioReplayBuffer

from Learner.Utils import get_masks


class BaseAgent(PrioReplayBuffer):
    def __init__(self, capacity: int, alpha: float, beta: float):
        super().__init__(capacity, alpha, beta)

    def sample_action(self, game: Game, state: T.Tensor, i_am_player: int, remember: bool) -> tuple[T.Tensor, T.Tensor]:
        pass

    def sample_village(self, game: Game, village_mask: T.Tensor, i_am_player: int):
        pass

    def sample_road(self, game: Game, road_mask: T.Tensor, i_am_player: int):
        pass

    def __repr__(self):
        return "BaseAgent"


class RandomAgent(BaseAgent):
    def __init__(self, capacity: int, alpha: float, beta: float) -> None:
        super().__init__(capacity, alpha, beta)

    def sample_action(self, game: Game, state: T.Tensor, i_am_player: int, remember: bool = True) -> tuple[T.Tensor, T.Tensor]:
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
    def __init__(self, q_net: GameNet, game: Game, capacity: int, alpha: float, beta: float) -> None:

        super().__init__(capacity, alpha, beta)
        self.q_net = q_net
        self.sparse_edge = game.board.state.edge_index.clone()
        self.empty_edge = T.zeros((self.sparse_edge.shape[1],), dtype=T.bool)
        self.empty_node = T.zeros((54,), dtype=T.bool)

        self.action = T.tensor((0, 0))
        self.raw_action = T.tensor((73, 73))

    def sample_action(self, game: Game, state: T.Tensor, i_am_player: int, remember: bool = True) -> tuple[T.Tensor, T.Tensor]:
        build_q: T.Tensor

        road_mask, village_mask = get_masks(game, i_am_player)

        if game.first_turn:
            if game.first_turn_village_switch:
                road_mask = self.empty_edge
            else:
                village_mask = self.empty_node

        mask = self.mask_to_dense(road_mask, village_mask)

        if len(mask.shape) == 3:
            mask = mask.squeeze()

        if mask.sum() == 0:
            if remember:
                self.add(state, mask, self.raw_action, 0, 0, game.episode, i_am_player)
            return self.action, self.raw_action
        with T.no_grad():
            q = self.q_net(state).detach()
        pass_q = q[0, -1, -1, i_am_player]
        q = q[0, :54, :54, i_am_player]
        q[~mask] = -T.inf
        build_q = q.max()

        if (pass_q > build_q) & (not game.first_turn):
            if remember:
                self.add(state, mask, self.raw_action, 0, 0, game.episode, i_am_player)
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
                    self.add(state, mask, build_action, 0, 0, game.episode, i_am_player)
                return T.tensor((1, index)), build_action

            elif build_action[0] == build_action[1]:
                if build_action[0] >= 54:
                    raise Exception("Non-node index returned for building settlement")
                if remember:
                    self.add(state, mask, build_action, 0, 0, game.episode, i_am_player)
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

    def __repr__(self) -> str:
        return "QAgent"

