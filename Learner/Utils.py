from dataclasses import dataclass
from typing import Tuple

import torch as T

from Environment import Game


class TensorDeque:
    def __init__(
            self,
            max_len: int = 100,
            queue_like: T.Tensor = None
    ) -> None:

        self.is_full: bool = False
        self._buffer: T.Tensor | None = None
        self._index: int = 0
        self._capacity = max_len

        if queue_like is None:
            self.shape = None
            self.dtype = None
        else:
            self.shape = queue_like.shape
            self.dtype = queue_like.dtype

        self.clear()

    def __len__(self):
        return self._capacity if self.is_full else self._index

    def __repr__(self, **kwargs):
        return self.to_tensor().__repr__(**kwargs)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if self.is_full:
                assert ((idx < self._capacity) & (idx >= -self._capacity)), f"index out of bounds, expected [{-self._capacity}, {self._capacity - 1}]"
            else:
                assert (idx < self._index) & (idx >= -self._index), f"index out of bounds, expected [{-self._index}, {self._index - 1}]"
            # Adjust idx to get the element in the circular buffer correctly
            adjusted_idx = (self._index + idx) % self._capacity
            return self._buffer[adjusted_idx]
        elif isinstance(idx, slice):
            # Handle slice indexing
            start, stop, step = idx.indices(self._capacity)
            if self.is_full:
                assert (start < self._capacity) & (
                        start >= -self._capacity), f"index out of bounds, expected lower bound [{-self._capacity}, {self._capacity - 1}]"
                assert (stop < self._capacity) & (
                        stop >= -self._capacity), f"index out of bounds, expected upper bound [{-self._capacity}, {self._capacity - 1}]"
            else:
                assert (start < self._index) & (
                        start >= -self._index), f"index out of bounds, expected lower bound [{-self._index}, {self._index - 1}]"
                assert (stop < self._index) & (
                        stop >= -self._index), f"index out of bounds, expected upper bound [{-self._index}, {self._index - 1}]"

            indices = [(self._index + i) % self._capacity for i in range(start, stop, step)]
            return self._buffer[indices]
        else:
            raise TypeError("Invalid index type.")

    @property
    def is_empty(self) -> bool:
        return (self._index == 0) & (not self.is_full)

    def clear(self) -> None:
        if self.shape is None:
            self._buffer = None
        else:
            self._buffer = T.zeros((self._capacity, *self.shape), dtype=self.dtype)

        self._index = 0
        self.is_full = False  # Flag to indicate whether the buffer has been filled at least once

    def append(self, element: T.Tensor | None) -> None:
        if self._buffer is None:
            if element is None:
                raise RuntimeError("Can't init unspecified TensorDeque with 'None' type element")
            self._buffer = T.zeros((self._capacity, *element.shape), dtype=element.dtype)
            self.shape = element.shape
            self.dtype = element.dtype

        if element is not None:
            assert self._buffer.shape[1:] == element.shape, f"Element shape mismatch, expected {self._buffer.shape[1:]}, got {element.shape}"
            self._buffer[self._index] = element

        self._increment()

    def to_tensor(self) -> T.Tensor:
        if self.is_full:
            # If the buffer is full, reorder such that the tensor is in the correct insertion order
            return T.cat((self._buffer[self._index:], self._buffer[:self._index]), dim=0)
        else:
            # If the buffer isn't full yet, only return the filled part, still in correct order
            return self._buffer[:self._index]

    def _increment(self) -> None:
        self._index += 1
        if self._index == self._capacity:
            self.is_full = True
            self._index = 0  # Reset index to start


@dataclass
class Transition:
    state: T.Tensor | None = None
    action: T.Tensor | None = None
    reward: T.Tensor | None = None
    lstm_state: T.Tensor | None = None
    lstm_cell: T.Tensor | None = None


def extract_attr(game: Game):
    node_x = game.board.state.x.clone()
    edge_x = game.board.state.edge_attr.clone()
    face_x = game.board.state.face_attr.clone()

    player_states = T.cat([ps.state.clone()[None, :] for ps in game.players], dim=1)
    node_x = T.cat((node_x, player_states.repeat((node_x.shape[0], 1))), dim=1)
    edge_x = T.cat((edge_x, player_states.repeat((edge_x.shape[0], 1))), dim=1)

    return node_x, edge_x, face_x


def sparse_face_matrix(face_index, to_undirected):
    n = face_index.size(0)  # Number of faces
    num_nodes_per_face = face_index.size(1)  # Should be 6
    face_indices = T.arange(54, 54 + n).repeat_interleave(num_nodes_per_face)

    node_indices = face_index.flatten()

    # Create the [2, K] matrix by stacking face_indices and node_indices
    connections = T.stack([node_indices, face_indices], dim=0)

    if to_undirected:
        connections = T.cat((connections, connections.flip(0)), dim=1)
    return connections


def sparse_misc_node(node_n, misc_n, to_undirected):
    node_range = T.arange(node_n + 1)
    sparse = T.stack((T.full_like(node_range, misc_n), node_range), dim=0)
    if to_undirected:
        sparse = T.cat((sparse, sparse.flip(0)), dim=1)
    return sparse


def preprocess_adj(adj, batch_size, add_self_loops):
    if add_self_loops:
        i = T.eye(adj.size(1)).to(adj.device)
        a_hat = adj[0] + i
    else:
        a_hat = adj[0]
    d_hat_diag = T.sum(a_hat, dim=1).pow(-0.5)
    d_hat = T.diag(d_hat_diag)
    adj_normalized = T.mm(T.mm(d_hat, a_hat), d_hat)
    return adj_normalized.repeat((batch_size, 1, 1))


def get_masks(game, i_am_player):
    road_mask = game.board.get_road_mask(i_am_player, game.players[i_am_player].hand, game.first_turn)
    village_mask = game.board.get_village_mask(i_am_player, game.players[i_am_player].hand, game.first_turn)
    return road_mask, village_mask


def get_cache_key(tensor: T.Tensor) -> Tuple:
    return tuple(tensor.numpy().flatten())
