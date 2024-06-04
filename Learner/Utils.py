from dataclasses import dataclass
from typing import Tuple, List

import torch as T


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


class TensorUtils:
    @staticmethod
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

    @staticmethod
    def sparse_misc_node(node_n, misc_n, to_undirected):
        node_range = T.arange(node_n + 1)
        sparse = T.stack((T.full_like(node_range, misc_n), node_range), dim=0)
        if to_undirected:
            sparse = T.cat((sparse, sparse.flip(0)), dim=1)
        return sparse

    @staticmethod
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

    @staticmethod
    def get_dense_masks(game, i_am_player):
        road_mask = game.board.get_road_mask(i_am_player, game.players[i_am_player].hand, game.first_turn)
        village_mask = game.board.get_village_mask(i_am_player, game.players[i_am_player].hand, game.first_turn)
        return road_mask, village_mask

    @staticmethod
    def get_cache_key(tensor: T.Tensor) -> Tuple:
        return tuple(tensor.numpy().flatten())

    @staticmethod
    def pairwise_isin(tensor_a: T.Tensor, tensor_b: T.Tensor) -> T.Tensor:
        # Step 1: Broadcasting and Comparison
        # We want each pair in tensor_a (2, N) to compare against every pair in tensor_b (2, M)
        # So, we reshape tensor_a to (2, N, 1) and tensor_b to (2, 1, M) to prepare for broadcasting
        tensor_a_exp = tensor_a.unsqueeze(2)  # Shape becomes (2, N, 1)
        tensor_b_exp = tensor_b.unsqueeze(1)  # Shape becomes (2, 1, M)
        # Now, perform element-wise comparison
        comparison = tensor_a_exp == tensor_b_exp  # Shape will be (2, N, M) after broadcasting
        # Step 2: Logical AND Operation
        # Check if both elements of the pair match
        pair_matches = comparison.all(dim=0)  # Collapse along the pair dimension, shape becomes (N, M)
        # Step 3: Aggregation
        # Determine if each pair in tensor_a matches with any pair in tensor_b
        matches = pair_matches.any(dim=1)

        return matches

    @staticmethod
    def gather_actions(values, indices):
        return values[
            T.arange(values.size(0)).unsqueeze(1),
            T.arange(values.size(1)),
            indices[:, :, 0],
            indices[:, :, 1]
        ]

    @staticmethod
    def propagate_rewards(gamma, rewards):
        """
        Propagates rewards backwards through a sequence.

        Args:
        - rewards (torch.Tensor): Tensor of shape [B, T] containing rewards,
          where B is batch size and T is sequence length.
        - gamma (float): Discount factor for future rewards.

        Returns:
        - torch.Tensor: Tensor of shape [B, T] with propagated rewards.
        """
        rewards = rewards.float()
        reversed_rewards = T.flip(rewards, dims=[1])
        updated_rewards = reversed_rewards.clone()

        for t in range(1, rewards.size(1)):
            updated_rewards[:, t] += gamma * updated_rewards[:, t - 1]

        propagated_rewards = T.flip(updated_rewards, dims=[1])

        return propagated_rewards

    @staticmethod
    def nn_sum(tensor: T.Tensor, dims: List[int]) -> T.Tensor:
        for dim in dims:
            tensor = tensor.sum(dim, keepdim=True)
        return tensor

    @staticmethod
    def get_batch_max(batched_tensor: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        b, s, n, _ = batched_tensor.shape
        max_act = T.empty((b, s, 2), dtype=T.long)
        max_q = T.empty((b, s)).to(batched_tensor.device)
        for i in range(b):
            for j in range(s):
                b_inds = T.argwhere(batched_tensor[i, j] == batched_tensor[i, j].max())
                if b_inds.numel() > 2:
                    b_inds = b_inds[T.randint(0, b_inds.shape[0], (1,))]
                max_act[i, j, :] = b_inds
                max_q[i, j] = batched_tensor[i, j].max()
        return max_act, max_q
