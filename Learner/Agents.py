from random import randint
import torch as T
import torch_geometric as pyg

import Learner.Nets


class Agent:
    def __init__(self):
        pass

    def sample_action(self, game, road_mask, village_mask, i_am_player):
        pass

    def sample_village(self, game, village_mask, i_am_player):
        pass

    def sample_road(self, game, road_mask, i_am_player):
        pass


class RandomAgent(Agent):
    def __init__(self):
        super().__init__()

    def sample_action(self, game, road_mask, village_mask, i_am_player):
        action_type = randint(0, 2)

        if action_type == 1:
            available_roads = road_mask.argwhere().squeeze()
            if available_roads.size()[0] == 0:
                return 0, 0
            index = T.randint(0, len(available_roads), (1,))
        elif action_type == 2:
            available_villages = village_mask.argwhere().squeeze()
            if available_villages.size()[0] == 0:
                return 0, 0
            index = T.randint(0, len(available_villages), (1,))
        else:
            index = 0

        return action_type, index

    def sample_village(self, game, village_mask, i_am_player):
        free_buildings = T.nonzero((game.board.state.x == 0).all(1)).squeeze()
        return T.randint(0, len(free_buildings), (1,))

    def sample_road(self, game, road_mask, i_am_player):
        free_roads = (game.board.state.edge_attr == 0).all(1)
        node_has_building = T.argwhere(game.board.state.x[:, i_am_player] > 0).squeeze()
        edge_has_building = T.isin(game.board.state.edge_index, node_has_building).any(0)
        free_edges = T.nonzero(free_roads & edge_has_building).squeeze()
        if free_edges.shape[0] == 0:
            Exception("No available edges!!")
        return free_edges[T.randint(0, len(free_edges), (1,))].item()


class QAgent(Agent):
    def __init__(self, q_net: Learner.Nets.GameNet, sparse_edge):
        super().__init__()
        self.q_net = q_net
        self.sparse_edge = sparse_edge
        self.empty_edge = T.zeros((self.sparse_edge.shape[1], ), dtype=T.bool)
        self.empty_node = T.zeros((54,), dtype=T.bool)

    def sample_action(self, game, road_mask, village_mask, i_am_player):
        build_q: T.Tensor

        mask = self.mask_to_dense(self.empty_edge, village_mask)
        if mask.sum() == 0:
            return 0, 0
        with T.no_grad():
            q = self.q_net(game).detach()
        pass_q = q[0, -1, -1, i_am_player]
        q = q[0, :54, :54, i_am_player]
        q[~mask] = -T.inf
        build_q = q.max()

        if pass_q > build_q:
            return 0, 0

        build_action = T.where(q == build_q)
        if build_action[0] != build_action[1]:
            bool_hit = (
                    (game.board.state.edge_index[0] == build_action[0])
                    & (game.board.state.edge_index[1] == build_action[1])
            )
            return 1, T.where(bool_hit)[0]

        if build_action[0] == build_action[1]:
            return 2, build_action[0]

    def sample_village(self, game, village_mask, i_am_player):
        mask = self.mask_to_dense(self.empty_edge, village_mask)
        with T.no_grad():
            q = self.q_net(game).detach()
        q = q[0, :54, :54, i_am_player]
        q[~mask] = -T.inf
        q = q[T.eye(q.shape[-1]).bool()]
        return q.argmax()


    def sample_road(self, game, road_mask, i_am_player):
        build_q: T.Tensor

        mask = self.mask_to_dense(road_mask, self.empty_node)[0]
        with T.no_grad():
            q = self.q_net(game).detach()
        q = q[0, :54, :54, i_am_player]
        q[~mask] = -T.inf
        build_q = q.max()
        build_action = T.argwhere(q == build_q).T
        build_action = build_action[:, 0]
        bool_hit = (
                (game.board.state.edge_index[0] == build_action[0])
                & (game.board.state.edge_index[1] == build_action[1])
        )
        return T.where(bool_hit)[0]

    def mask_to_dense(self, road_mask, village_mask):
        village_mask = T.diag_embed(village_mask).bool()
        if road_mask.sum() == 0:
            road_mask = T.zeros_like(village_mask, dtype=T.bool)
        else:
            road_mask = pyg.utils.to_dense_adj(self.sparse_edge[:, road_mask], max_num_nodes=village_mask.shape[-1]).bool()
        return village_mask + road_mask
