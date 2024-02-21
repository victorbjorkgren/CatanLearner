from random import randint
import torch as T
import torch_geometric as pyg

import Learner.Nets
# from Environment.Game import Game

from Learner.Utils import get_masks


class BaseAgent:
    def __init__(self):
        pass

    def sample_action(self, game, state, i_am_player: int):
        pass

    def sample_village(self, game, village_mask, i_am_player):
        pass

    def sample_road(self, game, road_mask, i_am_player):
        pass

    def __repr__(self):
        return "BaseAgent"


class RandomAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def sample_action(self, game, state, i_am_player: int) -> tuple[T.Tensor, T.Tensor]:
        road_mask, village_mask = get_masks(game, i_am_player)

        if game.first_turn:
            if game.first_turn_village_switch:
                index = self.sample_village(game, village_mask, i_am_player)
                if index is None:
                    breakpoint()
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
            Exception("Random Agent chose illegal action")

        breakpoint()

    def sample_village(self, game, village_mask, i_am_player) -> int:
        if village_mask.sum() > 0:
            available_villages = village_mask.argwhere().squeeze()
            if village_mask.sum() == 1:
                index = available_villages.item()
            else:
                index = available_villages[T.randint(0, len(available_villages), (1,))]
            return index
        else:
            Exception("Random Agent could not find house on round 1")

    def sample_road(self, game, road_mask, i_am_player) -> int:
        if road_mask.sum() > 0:
            available_roads = road_mask.argwhere().squeeze()
            if road_mask.sum() == 1:
                index = available_roads.item()
            else:
                index = available_roads[T.randint(0, len(available_roads), (1,))]
            return index
        else:
            Exception("Random Agent could not find road on round 1")

    def __repr__(self):
        return "RandomAgent"


class QAgent(BaseAgent):
    def __init__(self, q_net: Learner.Nets.GameNet, game):
        super().__init__()
        self.q_net = q_net
        self.sparse_edge = game.board.state.edge_index.clone()
        self.empty_edge = T.zeros((self.sparse_edge.shape[1],), dtype=T.bool)
        self.empty_node = T.zeros((54,), dtype=T.bool)

    def sample_action(self, game, state, i_am_player) -> tuple[T.Tensor, T.Tensor]:
        build_q: T.Tensor

        road_mask, village_mask = get_masks(game, i_am_player)

        if game.first_turn:
            if game.first_turn_village_switch:
                road_mask = self.empty_edge
            else:
                village_mask = self.empty_node


        mask = self.mask_to_dense(road_mask, village_mask)

        # TODO: Find root cause of over sized mask
        if len(mask.shape) == 3:
            mask = mask.squeeze()

        if mask.sum() == 0:
            return T.tensor((0, 0)), T.tensor((73, 73))
        with T.no_grad():
            q = self.q_net(state).detach()
        # TODO: Check why da fuq the agent doesn't build houses
        pass_q = q[0, -1, -1, i_am_player]
        q = q[0, :54, :54, i_am_player]
        q[~mask] = -T.inf
        build_q = q.max()

        if (pass_q > build_q) & (not game.first_turn):
            return T.tensor((0, 0)), T.tensor((73, 73))

        build_action = T.argwhere(q == build_q).cpu()
        if build_action.shape[0] > 1:
            build_action = build_action[T.randint(0, build_action.shape[0], (1,))].squeeze()
        elif build_action.shape[0] == 1:
            build_action = build_action.squeeze()
        else:
            Exception("Invalid Build Action in QAgent")

        if build_action[0] != build_action[1]:
            bool_hit = (
                    (game.board.state.edge_index[0] == build_action[0])
                    & (game.board.state.edge_index[1] == build_action[1])
            )
            index = T.argwhere(bool_hit).item()
            return T.tensor((1, index)), build_action

        elif build_action[0] == build_action[1]:
            if build_action[0] >= 54:
                breakpoint()
            return T.tensor((2, build_action[0])), build_action
        else:
            Exception("Invalid Build Action in QAgent")

    # def sample_village(self, game, village_mask, i_am_player):
    #     mask = self.mask_to_dense(self.empty_edge, village_mask)
    #     with T.no_grad():
    #         state = self.state_conv.get_dense(game)
    #         q = self.q_net(state).detach()
    #     q = q[0, :54, :54, i_am_player]
    #     q[~mask] = -T.inf
    #     q = q[T.eye(q.shape[-1]).bool()]
    #     return q.argmax()
    #
    # def sample_road(self, game, road_mask, i_am_player):
    #     build_q: T.Tensor
    #
    #     mask = self.mask_to_dense(road_mask, self.empty_node)[0]
    #     with T.no_grad():
    #         state = self.state_conv.get_dense(game)
    #         q = self.q_net(state).detach()
    #     q = q[0, :54, :54, i_am_player]
    #     q[~mask] = -T.inf
    #     build_q = q.max()
    #     build_action = T.argwhere(q == build_q).T
    #     build_action = build_action[:, 0]
    #     bool_hit = (
    #             (game.board.state.edge_index[0] == build_action[0])
    #             & (game.board.state.edge_index[1] == build_action[1])
    #     )
    #     return T.where(bool_hit)[0]

    def mask_to_dense(self, road_mask, village_mask):
        village_mask = T.diag_embed(village_mask).bool()
        if road_mask.sum() == 0:
            road_mask = T.zeros_like(village_mask, dtype=T.bool)
        else:
            road_mask = pyg.utils.to_dense_adj(self.sparse_edge[:, road_mask],
                                               max_num_nodes=village_mask.shape[-1]).bool()
        return village_mask + road_mask

    def __repr__(self):
        return "QAgent"

