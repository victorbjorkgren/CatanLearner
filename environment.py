import random
import torch
import torch_geometric
from torch_geometric.utils.convert import from_networkx
import torch_geometric.data as pyg_data

from HexGrid.HexGrid import make_hex_grid

###
# Board States:
#
# Face states: one hot encoded with dice number as value. Bandit is 1|0
# [Bandit, Brick, Grain, Ore, Lumber, Wool]
# Edge states: Has road or not
# [me, enemy0, ..., enemyK]
# Node states: village = 1, town = 2
# [me, enemy0, ..., enemyK]

###
# Player states:
#
# TODO: add cards etc.
# [Bricks, Grains, Ores, Lumbers, Wools]
#

BOARD_SIZE = 3
N_NODES = 54
N_TILES = 19
N_TILE_TYPES = 6

TILE_NUMBERS = [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11]
TILE_TYPES = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]  # index of tile type


# Extend PyG Data object to include faces
class ExtendedData(pyg_data.Data):
    def __init__(self, edge_index=None, x=None, edge_attr=None, face_index=None, face_attr=None):
        super(ExtendedData, self).__init__(edge_index=edge_index, x=x, edge_attr=edge_attr)
        self.face_index = face_index
        self.face_attr = face_attr

class Board:
    def __init__(self, n_players):
        grid = make_hex_grid(BOARD_SIZE)

        # init states
        node_states = torch.zeros((len(grid.nodes), n_players), dtype=torch.float)
        edge_states = torch.zeros((len(grid.edges) * 2, n_players), dtype=torch.float)
        face_states = torch.zeros((N_TILES, N_TILE_TYPES), dtype=torch.float)

        # fill tiles
        for i in range(len(TILE_TYPES)):
            face_states[i, TILE_TYPES[i]] = TILE_NUMBERS[i]
        shuffled_faces = torch.randperm(face_states.size(0))
        face_states = face_states[shuffled_faces]

        grid = from_networkx(grid)

        verts_per_row = [7, 9, 11, 11, 9, 7]
        face_per_row = [3, 4, 5, 4, 3]
        v_cum = [sum(verts_per_row[:i]) for i in range(0, len(verts_per_row)+1)]
        f_cum = [sum(face_per_row[:i]) for i in range(0, len(face_per_row)+1)]
        face_inds = torch.zeros((N_TILES, 6), dtype=torch.long)
        # row = 0
        for row in range(len(face_per_row)):
            for col in range(face_per_row[row]):

                if verts_per_row[row] < verts_per_row[row + 1]:
                    row_offset = 1
                elif verts_per_row[row] == verts_per_row[row + 1]:
                    row_offset = 0
                else:
                    row_offset = -1

                col_offset = max(0, col * 3 - 1)
                inds = [
                    v_cum[row] + 0 + col_offset,
                    v_cum[row] + 1 + col_offset,
                    v_cum[row] + 2 + col_offset,
                    v_cum[row + 1] + 0 + col_offset + row_offset,
                    v_cum[row + 1] + 1 + col_offset + row_offset,
                    v_cum[row + 1] + 2 + col_offset + row_offset
                ]
                face_inds[f_cum[row] + col, :] = torch.tensor(inds)

        self.state = ExtendedData(
            edge_index=grid.edge_index,
            x=node_states,
            edge_attr=edge_states,
            face_index=face_inds,
            face_attr=face_states
        )

    def update_edges(self, index, player):
        # Ensure edge_index is in COO format and edge_attr corresponds to it
        # Assuming edge_index is (2, E) and edge_attr is (E, *)

        # Find indices for (u, v) and (v, u)
        u, v = self.state.edge_index[:, index]
        uv_indices = (self.state.edge_index[0] == u) & (self.state.edge_index[1] == v)
        vu_indices = (self.state.edge_index[0] == v) & (self.state.edge_index[1] == u)

        # Update the attribute for both edges
        self.state.edge_attr[uv_indices, player] = 1
        self.state.edge_attr[vu_indices, player] = 1


    def can_build_village(self, node_id, player):
        adj_mask = (self.state.edge_index == node_id).any(0)
        edges_adj = self.state.edge_index[:, adj_mask]
        neighbor_nodes = edges_adj.unique()

        roads = self.state.edge_attr[adj_mask, player].nonzero().numel()
        buildings = self.state.x[neighbor_nodes, :].nonzero().numel()

        is_free = buildings == 0
        has_connection = roads != 0

        return is_free & has_connection

    def can_build_road(self, edge_id, player):
        edge_index = self.state.edge_index

        # Extract the specific edge nodes
        u, v = edge_index[:, edge_id]
        edge_self_mask = (edge_index == u).any(0) & (edge_index == v).any(0)
        edge_neighbor_mask = (edge_index == u).any(0) ^ (edge_index == v).any(0)

        edge_value_self = self.state.edge_attr[edge_self_mask, :]
        edge_value_neighbor = self.state.edge_attr[edge_neighbor_mask, player]

        # Check if any of these values are non-zero
        is_free = edge_value_self.nonzero().numel() == 0
        has_connection = edge_value_neighbor.nonzero().numel() > 0

        return is_free & has_connection


class Hand:
    def __init__(self):
        self.state = [0, 0, 0, 0, 0]
        self.points = 0

    def add(self, ind, n):
        self.state[ind] += n

    def sub(self, ind, n):
        self.state[ind] -= n

    def rob(self):
        if sum(self.state) > 7:
            for i in range(len(self.state)):
                self.state[i] -= self.state[i] // 2

    def sample_action(self):
        return 0, 0


class Game:
    def __init__(self, n_players):
        self.n_players = n_players
        self.reset()
        self.game_loop()

    def reset(self):
        self.board = Board(self.n_players)
        self.hands = [Hand() for _ in range(self.n_players)]

    def game_loop(self):
        current_player = 0
        while self.game_on():
            self.resource_step()
            player_done = False
            while(not player_done):
                act_type, index = self.hands[current_player].sample_action()
                player_done = self.take_action(act_type, current_player, index)
            current_player = abs(current_player - 1)

    def game_on(self):
        for hand in self.hands:
            if hand.points > 10:
                return False
        return True

    def resource_step(self):
        dice = random.randint(1, 6) + random.randint(1, 6)

        # Rob if 7
        if dice == 7:
            for hand in self.hands:
                hand.rob()

        # Find nodes attain resources
        face_hits = torch.argwhere(self.board.state.face_attr == dice)
        for hit in face_hits:
            node_hits = self.board.state.face_index[hit[0]].flatten()
            player_gains = self.board.state.x[node_hits].sum(0)
            resource = hit[1] - 1  # 0 is bandit for board only, not player
            if resource > 0:
                for i, gain in enumerate(player_gains):
                    self.hands[i].add(resource.item(), gain.item())

    def take_action(self, act_type, player, index):
        # act_types
        # 0 = pass
        # 1 = road
        # 2 = village
        # TODO: trade, draw card
        if act_type == 0:
            return True
        if act_type == 1:
            self.build_road(index, player)
            return False
        if act_type == 2:
            self.build_village(index, player)
            return False

    def build_road(self, index, player):
        # [Bricks, Grains, Ores, Lumbers, Wools]
        if self.board.can_build_road(index, player):
            if (
                    (self.hands[player].state[0] > 0)
                    & (self.hands[player].state[3] > 0)
            ):
                self.hands[player].sub(0, 1)
                self.hands[player].sub(3, 1)
                self.board.update_edges(index, player)
                return True
        return False


    def build_village(self, index, size, player):
        if self.board.can_build_village(index, player):
            if size == 1:
                # [Bricks, Grains, Ores, Lumbers, Wools]
                if (
                        (self.hands[player].state[0] > 0)
                        & (self.hands[player].state[1] > 0)
                        & (self.hands[player].state[3] > 0)
                        & (self.hands[player].state[4] > 0)
                ):
                    self.hands[player].sub(0, 1)
                    self.hands[player].sub(1, 1)
                    self.hands[player].sub(3, 1)
                    self.hands[player].sub(4, 1)
                    self.board.state.x[index, player] = size
                    self.hands[player].points += 1
                    return True
            if size == 2:
                # [Bricks, Grains, Ores, Lumbers, Wools]
                if (
                        (self.hands[player].state[1] > 2)
                        & (self.hands[player].state[2] > 3)
                ):
                    self.hands[player].sub(1, 2)
                    self.hands[player].sub(2, 3)
                    self.board.state.x[index, player] = size
                    self.hands[player].points += 1
                    return True
        return False

game = Game(2)
game.resource_step()