from typing import Any

import torch as T
from torch_geometric.utils.convert import from_networkx
import torch_geometric.data as pyg_data

from HexGrid.HexGrid import make_hex_grid

from .constants import *


class ExtendedData(pyg_data.Data):
    """Extend PyG Data object to include faces"""
    def __init__(self, edge_index=None, x=None, pos=None, edge_attr=None, face_index=None, face_attr=None):
        super(ExtendedData, self).__init__(edge_index=edge_index, x=x, pos=pos, edge_attr=edge_attr)
        self.face_index = face_index
        self.face_attr = face_attr


class Board:
    """
    Board States:
    Face states: one hot encoded with dice number as value. Bandit is 1|0
    [Bandit, Brick, Grain, Ore, Lumber, Wool]
    Edge states: Has road or not
    [me, enemy0, ..., enemyK]
    Node states: village = 1, town = 2
    [me, enemy0, ..., enemyK]
    """
    def __init__(self, n_players):
        self.n_players = n_players

        # init states
        grid = make_hex_grid(BOARD_SIZE)
        node_states = T.zeros((len(grid.nodes), n_players), dtype=T.float)
        edge_states = T.zeros((len(grid.edges) * 2, n_players), dtype=T.float)
        face_states = T.zeros((N_TILES, N_TILE_TYPES), dtype=T.float)

        # Fill Faces as tiles
        for i in range(len(TILE_TYPES)):
            face_states[i, TILE_TYPES[i]] = TILE_NUMBERS[i]
        shuffled_faces = T.randperm(face_states.size(0))
        face_states = face_states[shuffled_faces]

        # Make Faces and assign Nodes
        verts_per_row = [7, 9, 11, 11, 9, 7]
        face_per_row = [3, 4, 5, 4, 3]
        v_cum = [sum(verts_per_row[:i]) for i in range(0, len(verts_per_row) + 1)]
        f_cum = [sum(face_per_row[:i]) for i in range(0, len(face_per_row) + 1)]
        face_inds = T.zeros((N_TILES, 6), dtype=T.long)
        for row in range(len(face_per_row)):
            for col in range(face_per_row[row]):

                if verts_per_row[row] < verts_per_row[row + 1]:
                    row_offset = 1
                else:
                    row_offset = 0

                if row >= 3:
                    step_down_offset = 1
                else:
                    step_down_offset = 0

                col_offset = max(0, col * 2)
                inds = [
                    v_cum[row] + 0 + col_offset + step_down_offset,
                    v_cum[row] + 1 + col_offset + step_down_offset,
                    v_cum[row] + 2 + col_offset + step_down_offset,
                    v_cum[row + 1] + 0 + col_offset + row_offset,
                    v_cum[row + 1] + 1 + col_offset + row_offset,
                    v_cum[row + 1] + 2 + col_offset + row_offset
                ]
                face_inds[f_cum[row] + col, :] = T.tensor(inds)

        # Make PyG state
        grid = from_networkx(grid)

        # if rendering:
        pos = grid.pos
        # else:
        #     pos = None

        self.state = ExtendedData(
            edge_index=grid.edge_index,
            x=node_states,
            pos=pos,
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

    def can_build_village(self, node_id: int | T.Tensor, player: int, first_turn: bool = False) -> tuple[bool, T.Tensor]:
        if isinstance(node_id, T.Tensor):
            node_id = node_id.item()
        adj_mask = (self.state.edge_index == node_id).any(0)
        edges_adj = self.state.edge_index[:, adj_mask]
        neighborhood_nodes = edges_adj.unique()
        adjacent_nodes = neighborhood_nodes[neighborhood_nodes != node_id]

        roads = self.state.edge_attr[adj_mask, player].nonzero().numel()
        adjacent_buildings = self.state.x[adjacent_nodes, :].nonzero().numel()
        try:
            my_buildings = self.state.x[node_id, player]
        except:
            breakpoint()
        other_players_buildings = (self.state.x[node_id, :].sum() - my_buildings)

        is_free = (adjacent_buildings == 0) & (other_players_buildings == 0) & (my_buildings < 2)
        has_connection = roads != 0

        return is_free & (has_connection | first_turn), my_buildings

    def get_village_mask(self, player, hand, first_turn=False) -> T.Tensor:
        mask = T.zeros((self.state.num_nodes, ), dtype=T.bool)
        for node_id in range(self.state.num_nodes):
            can_build, size = self.can_build_village(node_id, player, first_turn)
            if can_build & (size == 0) & (hand >= T.tensor([1, 1, 0, 1, 1])).all():
                mask[node_id] = True
            if can_build & (size == 1) & (hand >= T.tensor([0, 2, 3, 0, 0])).all():
                mask[node_id] = True
        return mask

    def get_road_mask(self, player, hand, first_turn=False) -> T.Tensor:
        mask = T.zeros((self.state.num_edges,), dtype=T.bool)
        if (hand < T.tensor([1, 0, 0, 1, 0])).any():
            return mask
        for edge_id in range(self.state.num_edges):
            can_build = self.can_build_road(edge_id, player, first_turn)
            mask[edge_id] = can_build
        return mask

    def can_build_road(self, edge_id, player, first_turn=False):
        edge_index = self.state.edge_index

        # Extract the specific edge nodes
        u, v = edge_index[:, edge_id]
        edge_self_mask = (edge_index == u).any(0) & (edge_index == v).any(0)
        edge_neighbor_mask = (edge_index == u).any(0) ^ (edge_index == v).any(0)

        edge_value_self = self.state.edge_attr[edge_self_mask, :]
        edge_value_neighbor = self.state.edge_attr[edge_neighbor_mask, player]

        # Check if any of these values are non-zero
        is_free = edge_value_self.nonzero().numel() == 0
        if first_turn:
            has_connection = (self.state.x[u, player] > 0) | (self.state.x[v, player] > 0)
        else:
            has_connection = edge_value_neighbor.nonzero().numel() > 0

        return is_free & has_connection
