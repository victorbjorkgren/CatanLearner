from typing import Tuple

import torch
from torch import Tensor
from dataclasses import dataclass, fields
from typing import Optional
from typing import Type, TypeVar, List


T = TypeVar('T', bound='Holders')


@dataclass
class Holders:
    """
    Base class for recurrent Holders.
    These classes are @dataclasses holding either Tensors or other Holders.
    Implements Tensor functions to act on the entire recurrent tree.
    """
    def __getitem__(self, index):
        """Called with str to be dict-like or index, slice or tensor to be tensor-like"""
        if isinstance(index, str):
            return getattr(self, index)

        # assert index < self.tensor_size, 'Index out of range'
        state_args = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, Holders):
                state_args[f.name] = val[index]
            else:
                state_args[f.name] = val[index][None, ...]
        return type(self)(**state_args)

    def __iter__(self):
        num_elements = self.tensor_size[0]
        for i in range(num_elements):
            state_args = {}
            for f in fields(self):
                val = getattr(self, f.name)
                if isinstance(val, list):
                    if not isinstance(val[0], Tensor):
                        raise TypeError(f'Unexpected type {type(val[0])}')
                    if val[0].shape[1] == 1:
                        val = torch.cat(val, dim=1)
                    else:
                        val = torch.cat(val, dim=0)
                if isinstance(val, Tensor):
                    state_args[f.name] = val[i][None, ...]
                elif isinstance(val, Holders):
                    state_args[f.name] = val[i]
                else:
                    raise TypeError(f'Unexpected type {type(val)}')
            yield type(self)(**state_args)

    # @classmethod
    # def _auto_pad(cls, obj_list: List[T]):
    #     max_t = cls._list_max_t(obj_list)
    #     return cls.pad_t_dim(obj_list, max_t)

    # @classmethod
    # def pad_t_dim(cls: Type[T], obj_list: List[T], max_t: int) -> List[T]:
    #     out_list = []
    #     for field_ in fields(obj_list[0]):
    #         field_name = field_.name
    #         field_values = [getattr(obj, field_name) for obj in obj_list]
    #         if isinstance(field_values[0], Holders):
    #             field_class = type(field_values[0])
    #             out_list.append(field_class.pad_t_dim(field_values, max_t))
    #         else:
    #             out_list.append(TensorUtils.pad_tensor_t_dim(field_values, max_t))
    #     return out_list

    # @staticmethod
    # def _list_max_t(obj_list: List[T], dim) -> int:
    #     max_t = 0
    #     for obj in obj_list:
    #         max_t = max(max_t, obj._max_t(dim=dim))
    #     return max_t

    # def _max_t(self, dim) -> int:
    #     """Find max shape of the t dim (dim idx 1) of the tensors in a holder"""
    #     size = 0
    #     for field_ in fields(self):
    #         val = getattr(self, field_.name)
    #         if isinstance(val, Holders):
    #             t = val._max_t(dim)
    #         elif isinstance(val, Tensor):
    #             if val.ndim > dim:
    #                 t = val.shape[dim]
    #             else:
    #                 continue
    #         else:
    #             continue
    #         if t > size:
    #             size = t
    #     return size

    @classmethod
    def concat(cls: Type[T], obj_list: List[T], dim: int, pad_dim=None, max_t=0) -> T:
        # if pad_dim is not None:
        #     max_t_ = cls._list_max_t(obj_list, pad_dim)
        #     if max_t < max_t_:
        #         max_t = max_t_
        result_tensors = {}
        for field_ in fields(obj_list[0]):
            field_name = field_.name
            field_values = []
            for obj in obj_list:
                field_value = getattr(obj, field_name)
                # if pad_dim is not None and isinstance(field_value, Tensor):
                #     field_value = TensorUtils.pad_tensor_dim(field_value, max_t)
                field_values.append(field_value)
            if isinstance(field_values[0], Holders):
                if 'not_stackable' in field_.metadata.keys():
                    result_tensors[field_name] = field_values
                else:
                    field_class = type(field_values[0])
                    result_tensors[field_name] = field_class.concat(field_values, dim, pad_dim, max_t)
            else:
                if 'not_stackable' in field_.metadata.keys():
                    result_tensors[field_name] = field_values
                else:
                    field_values = TensorUtils.pad_tensor_list(field_values)
                    result_tensors[field_name] = torch.cat(field_values, dim=dim)
        return cls(**result_tensors)

    @classmethod
    def stack(cls: Type[T], obj_list: List[T], dim: int, pad_dim=None, max_t=0) -> T:
        # if pad_dim is not None:
        #     max_t_ = cls._list_max_t(obj_list, pad_dim)
        #     if max_t < max_t_:
        #         max_t = max_t_
        result_tensors = {}
        for field_ in fields(obj_list[0]):
            field_name = field_.name
            field_values = []
            for obj in obj_list:
                field_value = getattr(obj, field_name)
                # will_pad = pad_dim is not None and field_value.ndim > pad_dim
                # is_tensor = isinstance(field_value, Tensor)
                # is_stackable = 'not_stackable' not in field_.metadata.keys()
                # if will_pad and is_tensor and is_stackable:
                #     field_value = TensorUtils.pad_tensor_dim(field_value, max_t)
                field_values.append(field_value)
            if isinstance(field_values[0], Holders):
                field_class = type(field_values[0])
                result_tensors[field_name] = field_class.stack(field_values, dim, pad_dim, max_t)
            else:
                if 'not_stackable' in field_.metadata.keys():
                    result_tensors[field_name] = field_values
                else:
                    field_values = TensorUtils.pad_tensor_list(field_values)
                    result_tensors[field_name] = torch.stack(field_values, dim=dim)
        return cls(**result_tensors)

    def deep_copy(self):
        kwargs = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, Tensor):
                kwargs[f.name] = val.clone()
            elif isinstance(val, Holders):
                kwargs[f.name] = val.deep_copy()
        return type(self)(**kwargs)

    def to(self, device):
        for field_ in fields(self):
            val = getattr(self, field_.name)
            if isinstance(val, Tensor):
                setattr(self, field_.name, val.to(device))
            elif isinstance(val, Holders):
                val.to(device)
            elif isinstance(val, list):
                pass
            else:
                raise TypeError(f'Unexpected type {type(val)}')
        return self

    def assert_dim(self, dims: List[int] = None):
        for field_ in fields(self):
            val = getattr(self, field_.name)
            if isinstance(val, Tensor):
                assert val.ndim < 5, f'Tensor dim {val.ndim} must be less than 5 for mps'
                if dims is not None:
                    for i in range(len(dims) - 1):  # Never checking f dim
                        assert val.shape[i] == dims[i], f'Tensor shape is {val.shape[i]} expected {dims[i]}'
            elif isinstance(val, Holders):
                val.detach()
            else:
                raise TypeError(f'Unexpected type {type(val)}')

    def detach(self):
        for field_ in fields(self):
            val = getattr(self, field_.name)
            if isinstance(val, Tensor):
                setattr(self, field_.name, val.detach())
            elif isinstance(val, Holders):
                val.detach()
            elif isinstance(val, list):
                setattr(self, field_.name, [e.detach() for e in val])
            else:
                raise TypeError(f'Unexpected type {type(val)}')
        return self

    def squeeze(self, dim):
        for field_ in fields(self):
            val = getattr(self, field_.name)
            if isinstance(val, Tensor):
                setattr(self, field_.name, val.squeeze(dim))
            elif isinstance(val, Holders):
                val.squeeze(dim)
            else:
                raise TypeError(f'Unexpected type {type(val)}')
        return self

    def unsqueeze(self, dim):
        for field_ in fields(self):
            val = getattr(self, field_.name)
            if isinstance(val, Tensor):
                setattr(self, field_.name, val.unsqueeze(dim))
            elif isinstance(val, Holders):
                val.unsqueeze(dim)
            else:
                raise TypeError(f'Unexpected type {type(val)}')
        return self

    def dim(self):
        for field_ in fields(self):
            return getattr(self, field_.name).ndim
        return self

    def requires_grad_(self):
        for field_ in fields(self):
            val = getattr(self, field_.name)
            if isinstance(val, Tensor):
                if val.dtype == torch.float or val.dtype == torch.half or val.dtype == torch.double:
                    val.requires_grad_()
                setattr(self, field_.name, val)
            elif isinstance(val, Holders):
                val.requires_grad_()
        return self

    @property
    def tensor_size(self) -> torch.Size:
        f = fields(self)
        assert len(f) > 0, "Called Holders.tensor_size on empty Holder"
        first_field_value = getattr(self, f[0].name)
        if isinstance(first_field_value, Holders):
            return first_field_value.tensor_size
        elif isinstance(first_field_value, list):
            if isinstance(first_field_value[0], Tensor):
                shape = list(first_field_value[0].shape)
            elif isinstance(first_field_value[0], Holders):
                shape = list(first_field_value[0].tensor_size)
            else:
                raise TypeError(f'Unexpected type {type(first_field_value[0])}')
            if shape[1] == 1:
                shape[1] = len(first_field_value)
            else:
                shape[0] = len(first_field_value)
            return torch.Size(shape)
        elif isinstance(first_field_value, Tensor):
            return first_field_value.shape
        else:
            raise TypeError(f'Unexpected type {type(first_field_value)}')


class TensorDeque:
    def __init__(
            self,
            max_len: int = 100,
            queue_like: Tensor = None
    ) -> None:

        self.is_full: bool = False
        self._buffer: Tensor | None = None
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
                assert ((idx < self._capacity) & (
                            idx >= -self._capacity)), f"index out of bounds, expected [{-self._capacity}, {self._capacity - 1}]"
            else:
                assert (idx < self._index) & (
                            idx >= -self._index), f"index out of bounds, expected [{-self._index}, {self._index - 1}]"
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
            self._buffer = torch.zeros((self._capacity, *self.shape), dtype=self.dtype)

        self._index = 0
        self.is_full = False  # Flag to indicate whether the buffer has been filled at least once

    def append(self, element: Tensor | None) -> None:
        if self._buffer is None:
            if element is None:
                raise RuntimeError("Can't init unspecified TensorDeque with 'None' type element")
            self._buffer = torch.zeros((self._capacity, *element.shape), dtype=element.dtype)
            self.shape = element.shape
            self.dtype = element.dtype

        if element is not None:
            assert self._buffer.shape[
                   1:] == element.shape, f"Element shape mismatch, expected {self._buffer.shape[1:]}, got {element.shape}"
            self._buffer[self._index] = element

        self._increment()

    def to_tensor(self) -> Tensor:
        if self.is_full:
            # If the buffer is full, reorder such that the tensor is in the correct insertion order
            return torch.cat((self._buffer[self._index:], self._buffer[:self._index]), dim=0)
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
    state: Tensor | None = None
    action: Tensor | None = None
    reward: Tensor | None = None
    lstm_state: Tensor | None = None
    lstm_cell: Tensor | None = None


class TensorUtils:
    @staticmethod
    def pop_elements(data_tensor, bool_tensor):
        # Initialize new tensor for storing the results
        result = []

        # Iterate through each row
        for i in range(data_tensor.shape[0]):
            if bool_tensor[i]:
                # Pop from the left
                result.append(data_tensor[i, 1:])
            else:
                # Pop from the right
                result.append(data_tensor[i, :-1])

        # Stack the results to maintain the same tensor structure
        result_tensor = torch.stack(result)
        return result_tensor

    @staticmethod
    def sparse_face_matrix(face_index, to_undirected):
        n = face_index.size(0)  # Number of faces
        num_nodes_per_face = face_index.size(1)  # Should be 6
        face_indices = torch.arange(54, 54 + n).repeat_interleave(num_nodes_per_face)

        node_indices = face_index.flatten()

        # Create the [2, K] matrix by stacking face_indices and node_indices
        connections = torch.stack([node_indices, face_indices], dim=0)

        if to_undirected:
            connections = torch.cat((connections, connections.flip(0)), dim=1)
        return connections

    @staticmethod
    def sparse_game_node(node_n, misc_n, to_undirected):
        node_range = torch.arange(node_n + 1)
        sparse = torch.stack((torch.full_like(node_range, misc_n), node_range), dim=0)
        if to_undirected:
            sparse = torch.cat((sparse, sparse.flip(0)), dim=1)
        return sparse

    @staticmethod
    def preprocess_adj(adj, batch_size, add_self_loops):
        if add_self_loops:
            i = torch.eye(adj.size(1)).to(adj.device)
            a_hat = adj[0] + i
        else:
            a_hat = adj[0]
        d_hat_diag = torch.sum(a_hat, dim=1).pow(-0.5)
        d_hat = torch.diag(d_hat_diag)
        adj_normalized = torch.mm(torch.mm(d_hat, a_hat), d_hat)
        return adj_normalized.repeat((batch_size, 1, 1))

    @staticmethod
    def get_dense_masks(game, i_am_player):
        road_mask = game.board.get_road_mask(i_am_player, game.players[i_am_player].hand, game.first_turn)
        village_mask = game.board.get_village_mask(i_am_player, game.players[i_am_player].hand, game.first_turn)
        return road_mask, village_mask

    @staticmethod
    def get_cache_key(tensor: Tensor) -> Tuple:
        return tuple(tensor.numpy().flatten())

    @staticmethod
    def pairwise_isin(tensor_a: Tensor, tensor_b: Tensor) -> (Tensor, Tensor):
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
        indices = pair_matches.argwhere()

        return matches, indices

    @staticmethod
    def gather_actions(q_values: Tensor, trade_q_values: Tensor, indices: Tensor, was_trade: Optional[Tensor]) -> Tensor:
        if was_trade is None:
            was_trade = torch.zeros(indices.shape[:2], dtype=torch.bool)

        q_out = torch.empty(indices.shape[:2]).to(q_values.device)
        was_trade = was_trade.cpu()
        was_trade_range = torch.arange(was_trade.sum())
        was_trade_zero = torch.zeros(was_trade.sum()).long()
        was_trade_one = torch.ones(was_trade.sum()).long()
        give_q = trade_q_values[was_trade][was_trade_range, was_trade_zero, indices[was_trade][:, 0]]
        get_q = trade_q_values[was_trade][was_trade_range, was_trade_one, indices[was_trade][:, 1]]

        not_was_trade_range = torch.arange((~was_trade).sum())
        build_q = q_values[~was_trade][not_was_trade_range, indices[~was_trade][:, 0], indices[~was_trade][:, 1]]
        if was_trade.sum() > 0:
            q_out[was_trade] = give_q + get_q
        if (~was_trade).sum() > 0:
            q_out[~was_trade] = build_q
        return q_out

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
        reversed_rewards = torch.flip(rewards, dims=[1])
        updated_rewards = reversed_rewards.clone()

        for t in range(1, rewards.size(1)):
            updated_rewards[:, t] += gamma * updated_rewards[:, t - 1]

        propagated_rewards = torch.flip(updated_rewards, dims=[1])

        return propagated_rewards

    @staticmethod
    def nn_sum(tensor: Tensor, dims: List[int]) -> Tensor:
        for dim in dims:
            tensor = tensor.sum(dim, keepdim=True)
        return tensor

    @staticmethod
    def nn_mean(tensor: Tensor, dims: List[int]) -> Tensor:
        for dim in dims:
            tensor = tensor.mean(dim, keepdim=True)
        return tensor

    @staticmethod
    def get_batch_max(batched_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        b, s, n, _ = batched_tensor.shape
        max_act = torch.empty((b, s, 2), dtype=torch.long)
        max_q = torch.empty((b, s)).to(batched_tensor.device)
        for i in range(b):
            for j in range(s):
                b_inds = torch.argwhere(batched_tensor[i, j] == batched_tensor[i, j].max())
                if b_inds.numel() > 2:
                    b_inds = b_inds[torch.randint(0, b_inds.shape[0], (1,))]
                max_act[i, j, :] = b_inds
                max_q[i, j] = batched_tensor[i, j].max()
        return max_act, max_q

    @staticmethod
    def non_zero_mean(tensor, dim):
        mean = tensor.sum(dim) / tensor.count_nonzero(dim)
        return mean.unsqueeze(1)

    @staticmethod
    def non_zero_std(tensor, dim):
        count_nonzero = tensor.count_nonzero(dim)
        sum_nonzero = tensor.sum(dim)
        mean_nonzero = sum_nonzero / count_nonzero
        mean_nonzero = mean_nonzero.unsqueeze(1)

        squared_diff = (tensor - mean_nonzero) ** 2
        squared_diff = squared_diff * (tensor != 0).float()
        sum_squared_diff = squared_diff.sum(dim)

        variance_nonzero = sum_squared_diff / count_nonzero
        std_nonzero = torch.sqrt(variance_nonzero)
        return std_nonzero.unsqueeze(1)

    @staticmethod
    def advantage_estimation(r_t: Tensor, value_t: Tensor, done: Tensor, seq_lens: Tensor, gamma: float) -> tuple[Tensor, Tensor]:
        """Computes truncated generalized advantage estimates for a sequence length k.

        The advantages are computed in a backwards fashion according to the equation:
        Âₜ = δₜ + (γλ) * δₜ₊₁ + ... + ... + (γλ)ᵏ⁻ᵗ⁺¹ * δₖ₋₁
        where δₜ = rₜ + γₜ * v(sₜ₊₁) - v(sₜ).

        See Proximal Policy Optimization Algorithms, Schulman et al.:
        https://arxiv.org/abs/1707.06347

        Args:
          r_t: Sequence of rewards at times [0, k]
          value_t: Sequence of values under π at times [0, k]

        Returns:
          Multi-step truncated generalized advantage estimation at times [0, k-1].
        """
        assert value_t.ndim == 2
        assert r_t.ndim == 2
        assert done.ndim == 1
        LAMBDA = 0.95
        b, t = value_t.shape

        # value_t_ = torch.zeros_like(value_t)
        # value_tp1_ = torch.zeros_like(value_t)
        mask = (torch.arange(t)[None, :] >= seq_lens[:, None].cpu()).to(r_t.device)
        value_t[mask] = 0

        # value_tp1_ = torch.zeros(b, t - 1, device=r_t.device)
        # value_tp1_[done, :] = torch.cat((value_t[done, 2:], torch.zeros(done.sum(), 1, device=r_t.device)), dim=1)
        # value_tp1_[~done, :] = value_t[~done, 1:]
        value_tp1_ = torch.cat((value_t[:, 1:], torch.zeros(done.sum(), 1, device=r_t.device)), dim=1)

        # value_t_ = torch.zeros(b, t - 1, device=r_t.device)
        # value_t_[done, :] = value_t[done, 1:]
        # value_t_[~done, :] = value_t[~done, :-1]
        value_t_ = value_t.clone()

        # r_t = TensorUtils.pop_elements(r_t, done)

        delta_value = gamma * value_tp1_ - value_t_

        delta_t = r_t + delta_value

        advantage_t = torch.zeros_like(delta_t, dtype=torch.float32)

        gae_t = torch.zeros(b, device=r_t.device)
        t = seq_lens.clone() - 2
        for _ in range(delta_t.shape[1]):
            seq_mask = t >= 0
            t_ = t[seq_mask]
            gae_t[seq_mask] = delta_t[seq_mask, t_] + gamma * LAMBDA * gae_t[seq_mask]
            advantage_t[seq_mask, t_] = gae_t[seq_mask]
            t -= 1

        return_t = advantage_t + value_t_
        advantage_t = (advantage_t - TensorUtils.non_zero_mean(advantage_t, 1)) / (TensorUtils.non_zero_std(advantage_t, 1) + 1e-8)
        advantage_t[mask] = 0
        return advantage_t, return_t

    @classmethod
    def pad_tensor_list(cls, tensor_list: List[Tensor]) -> List[Tensor]:
        pad_tensor_list = []
        max_shape = torch.tensor(tensor_list[0].shape)
        for e in tensor_list:
            max_shape = torch.max(max_shape, torch.tensor(e.shape))
        for obj in tensor_list:
            pad = torch.zeros(*max_shape, dtype=obj.dtype, device=obj.device,
                              requires_grad=obj.requires_grad)
            idx = tuple(slice(0, size) for size in obj.shape)
            pad[*idx] = obj
            pad_tensor_list.append(pad)
        return pad_tensor_list

    @staticmethod
    def pad_tensor_dim(tensor, size, dim=1):
        if tensor.ndim <= dim:
            return tensor
        if tensor.shape[dim] < size:
            target_shape = list(tensor.shape)
            target_shape[dim] = size
            pad = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device,
                              requires_grad=tensor.requires_grad)
            idx = tuple(slice(0, size) for size in tensor.shape)
            pad[*idx] = tensor
            return pad
        else:
            return tensor


class LinearSchedule:
    """Linear schedule, for changing constants during agent training."""

    def __init__(self, begin_value, end_value, begin_t, end_t=None, decay_steps=None):
        if (end_t is None) == (decay_steps is None):
            raise ValueError('Exactly one of end_t, decay_steps must be provided.')
        self._decay_steps = decay_steps if end_t is None else end_t - begin_t
        self._begin_t = begin_t
        self._begin_value = begin_value
        self._end_value = end_value

    def __call__(self, t_):
        """Implements a linear transition from a beginning to an end value."""
        frac = min(max(t_ - self._begin_t, 0), self._decay_steps) / self._decay_steps
        return (1 - frac) * self._begin_value + frac * self._end_value
