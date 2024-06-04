from random import randint
import torch

class RandomAgent:
    def __init__(self):
        pass

    def sample_action(self, board, players, i_am_player):
        action_type = randint(0, 2)

        if action_type == 1:
            index = randint(0, board.state.num_edges // 2 - 1)
        elif action_type == 2:
            index = randint(0, board.state.num_nodes - 1)
        else:
            index = 0

        return action_type, index

    def sample_building(self, board, players, i_am_player):
        free_buildings = torch.nonzero((board.state.x == 0).all(1)).squeeze()
        return torch.randint(0, len(free_buildings), (1,))

    def sample_road(self, board, players, i_am_player):
        free_roads = (board.state.edge_attr == 0).all(1)
        node_has_building = torch.argwhere(board.state.x[:, i_am_player] > 0).squeeze()
        edge_has_building = torch.isin(board.state.edge_index, node_has_building).any(0)
        free_edges = torch.nonzero(free_roads & edge_has_building).squeeze()
        if free_edges.shape[0] == 0:
            Exception("No available edges!!")
        return free_edges[torch.randint(0, len(free_edges), (1,))].item()
