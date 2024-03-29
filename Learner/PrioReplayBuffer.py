import os
import pickle

import numpy as np
import torch as T


def assert_capacity(n):
    # Check that n is an integer greater than 0 and n & (n - 1) == 0
    assert isinstance(n, int) and n > 0 and n & (n - 1) == 0, "Value is not a power of 2"


class PrioReplayBuffer:
    _capacity: int
    _alpha: float
    _beta: float
    _max_priority: float
    _size: int
    data: dict
    _next_idx: int

    def __init__(self,
                 capacity: int,
                 max_seq_len: int,
                 alpha: float,
                 beta: float,
                 save_interval: int = 10_000
                 ) -> None:
        assert_capacity(capacity)

        if self.load():
            return

        self._capacity = capacity
        self._alpha = alpha
        self._beta = beta

        self._save_interval = save_interval
        self._save_countdown = save_interval

        self._max_priority = 1.
        self._sum_priority = 0.

        # TODO: Make board size invariant
        # TODO: Fix Magic Numbers
        self.data = {
            'state': T.zeros((capacity, max_seq_len, 74, 74, 16), dtype=T.float),
            'seq_len': T.zeros((capacity,), dtype=T.long),
            'action': T.zeros((capacity, max_seq_len, 2), dtype=T.long),
            'was_trade': T.zeros((capacity, max_seq_len), dtype=T.bool),
            'reward': T.zeros((capacity, max_seq_len), dtype=T.float),
            'lstm_state': T.zeros((capacity, max_seq_len, 32), dtype=T.float),
            'lstm_cell': T.zeros((capacity, max_seq_len, 32), dtype=T.float),
            'done': T.zeros((capacity,), dtype=T.bool),
            'episode': T.zeros((capacity,), dtype=T.long),
            'player': T.zeros((capacity,), dtype=T.long),
            'prio': np.zeros((capacity,), dtype=float)
        }

        self._buffer = {}
        self._req_data_keys = set(self.data.keys())
        self._req_data_keys.remove('prio')

        self._next_idx = 0
        self._size = 0

    def add(self,
            state: T.Tensor,
            action: T.Tensor,
            was_trade: T.Tensor,
            reward: T.Tensor,
            td_error: T.Tensor,
            lstm_state: T.Tensor,
            lstm_cell: T.Tensor,
            done: bool,
            episode: int,
            player: int
            ) -> None:

        if self.is_full:
            # idx = self.min_prio_idx
            if td_error.item() < self.data['prio'][self.min_prio_idx]:
                return
        # else:
        idx = self._next_idx
        self._next_idx = (idx + 1) % self._capacity

        state = state.squeeze()
        reward = reward.squeeze()
        was_trade = was_trade.squeeze()

        self.data['state'][idx, :len(state)] = state
        self.data['action'][idx, :len(action)] = action
        self.data['was_trade'][idx, :len(was_trade)] = was_trade
        self.data['reward'][idx, :len(reward)] = reward
        self.data['lstm_state'][idx, :len(lstm_state)] = lstm_state
        self.data['lstm_cell'][idx, :len(lstm_cell)] = lstm_cell
        self.data['seq_len'][idx] = len(state)
        self.data['done'][idx] = done
        self.data['episode'][idx] = episode
        self.data['player'][idx] = player
        self.data['prio'][idx] = min(td_error.item(), self._max_priority) ** self._alpha

        self._size = min(self._capacity, self._size + 1)

        # self.save_test()

    def sample(self, n):
        prob = self.data['prio'] / self.data['prio'].sum()
        try:
            sample_inds = np.random.choice(self._capacity, n, p=prob, replace=False)
        except ValueError:
            sample_inds = np.random.choice(self._capacity, n, replace=False)

        weights = (self._size * prob[sample_inds]) ** (-self._beta)
        weights = weights.clip(0, 1)

        samples = {
            'inds': sample_inds,
            'weights': weights
        }

        return samples

    def update_prio(self, ind: T.Tensor, prio: np.array):
        clipped_prio = prio.clip(max=self._max_priority)
        self.data['prio'][ind] = clipped_prio

    def save_test(self):
        if self._save_countdown <= 0:
            self.save()
            self._save_countdown = self._save_interval
        else:
            self._save_countdown -= 1

    def save(self):
        os.makedirs('./ReplayBuffer/', exist_ok=True)
        with open(f'./ReplayBuffer/ReplayBuffer.pkl', 'wb') as f:
            pickle.dump(self, f)

    def load(self) -> bool:
        try:
            with open(f'./ReplayBuffer/ReplayBuffer.pkl', 'rb') as f:
                temp_self = pickle.load(f)
                self.__dict__.update(temp_self.__dict__)
                return True
        except FileNotFoundError:
            print('No buffer found to load - init fresh buffer')
            return False

    @property
    def mem_size(self):
        return self._size

    @property
    def min_prio_idx(self):
        return self.data["prio"][self.data["prio"].nonzero()].argmin()

    @property
    def reward_sum(self):
        return (self.data['reward'] > 0).sum()

    @property
    def is_full(self):
        return self._capacity == self._size

    @property
    def save_interval(self):
        return self._save_interval

    @save_interval.setter
    def save_interval(self, new_val):
        self._save_countdown = new_val
        self._save_interval = new_val
