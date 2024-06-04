import numpy as np
import torch as T
import pickle


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

    def __init__(self, capacity: int, alpha: float, beta: float, save_interval: int = 1000):
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

        # TODO: Make board size and n_player invariant
        # TODO: Fix Magic Numbers
        self.data = {
            'state': T.zeros((capacity, 74, 74, 14)),
            'state_mask': T.zeros((capacity, 74, 74, 1), dtype=T.bool),
            'action': T.zeros((capacity, 2), dtype=T.long),
            'reward': T.zeros((capacity, 2)),
            'done': T.zeros((capacity,)),
            'episode': T.zeros((capacity,)),
            'episode_tick': T.zeros((capacity,)),
            'prio': np.zeros((capacity,))
        }

        # self._buffer = {}
        # self._req_data_keys = set(self._data.keys())
        # self._req_data_keys.remove('prio')

        self._next_idx = 0
        self._size = 0

    def add(self, state, state_mask, action, reward, done, episode):
        if self.is_full:
            idx = self.min_prio_idx
        else:
            idx = self._next_idx
            self._next_idx = (idx + 1) % self._capacity

        # TODO: Find root cause of this issue
        if len(action.shape) > 1:
            action = action.squeeze()

        self.data['state'][idx] = state
        self.data['state_mask'][idx] = state_mask
        self.data['action'][idx] = action.long()
        self.data['reward'][idx] = reward
        self.data['done'][idx] = done
        self.data['episode'] = episode
        self.data['prio'][idx] = self._max_priority ** self._alpha

        self._size = min(self._capacity, self._size + 1)

    def sample(self, n):
        prob = self.data['prio'] / self.data['prio'].sum()

        sample_inds = np.random.choice(self._capacity, n, p=prob)

        weights = (self._size * prob[sample_inds]) ** (-self._beta)
        weights /= weights.max()

        samples = {
            'inds': sample_inds,
            'weights': weights
        }
        # for k, v in self.data.items():
        #     samples[k] = v[sample_inds]

        return samples

    def update_prio(self, ind: T.Tensor, prio: np.array):
        clipped_prio = prio.clip(max=self._max_priority)
        self._sum_priority += clipped_prio.sum() - self.data['prio'][ind].sum()
        self.data['prio'][ind] = clipped_prio

    # def add_to_buffer(self, key, value) -> bool:
    #     """returns True if buffer was flushed"""
    #     if key in self._buffer:
    #         raise KeyError("Tried to overwrite buffer value")
    #
    #     self._buffer[key] = value
    #
    #     if self._req_data_keys.issubset(self._buffer.keys()):
    #         self.flush_buffer()
    #         return True
    #     return False
    #
    # def flush_buffer(self):
    #     self.add(
    #         self._buffer['state'],
    #         self._buffer['state_mask'],
    #         self._buffer['action'],
    #         self._buffer['new_state'],
    #         self._buffer['new_state_mask'],
    #         self._buffer['reward'],
    #         self._buffer['done'],
    #         # self._player
    #     )
    #     self._buffer.clear()

    def save_test(self):
        if self._save_countdown <= 0:
            self.save()
            self._save_countdown = self._save_interval
        else:
            self._save_countdown -= 1

    def save(self):
        with open('./ReplayBuffer/_buffer.pkl', 'wb') as f:
            pickle.dump(self, f)

    def load(self) -> bool:
        try:
            with open('./ReplayBuffer/_buffer.pkl', 'rb') as f:
                temp_self = pickle.load(f)
                self.__dict__.update(temp_self.__dict__)
                return True
        except FileNotFoundError:
            print('No buffer found to load - init fresh buffer')
            return False

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
    def priority_sum(self) -> float:
        return self._sum_priority

    @property
    def save_interval(self):
        return self._save_interval

    @save_interval.setter
    def save_interval(self, new_val):
        self._save_countdown = new_val
        self._save_interval = new_val