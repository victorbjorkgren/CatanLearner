import numpy as np
import torch as T


def assert_capacity(n):
    # Check that n is an integer greater than 0 and n & (n - 1) == 0
    assert isinstance(n, int) and n > 0 and n & (n - 1) == 0, "Value is not a power of 2"


class PrioReplayBuffer:
    def __init__(self, capacity, alpha, beta):
        assert_capacity(capacity)

        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta

        # self.priority_sum = [0] * (2 * capacity)
        # self.priority_min = [float('inf')] * (2 * capacity)

        self.max_priority = 1.

        # TODO: Make board size and n_player invariant
        # TODO: Fix Magic Numbers
        self.data = {
            'state': T.zeros((capacity, 74, 74, 14)),
            'action': T.zeros((capacity, 2), dtype=T.long),
            'new_state': T.zeros((capacity, 74, 74, 14)),
            'reward': T.zeros((capacity, 2)),
            'done': T.zeros((capacity,)),
            'player': T.zeros((capacity,), dtype=T.long),
            'prio': np.zeros((capacity,))
        }

        self.next_idx = 0
        self.size = 0

    def add(self, state, action, new_state, reward, done, i_am_player):
        if self.is_full:
            idx = self.min_prio_idx
        else:
            idx = self.next_idx
            self.next_idx = (idx + 1) % self.capacity

        # TODO: Find root cause of this issue
        if len(action.shape) > 1:
            action = action.squeeze()

        self.data['state'][idx] = state
        self.data['action'][idx] = action.long()
        self.data['new_state'][idx] = new_state
        self.data['reward'][idx] = reward
        self.data['done'][idx] = done
        self.data['player'][idx] = i_am_player
        self.data['prio'][idx] = self.max_priority ** self.alpha

        self.size = min(self.capacity, self.size + 1)

    def sample(self, n):
        prob = self.data['prio'] / self.data['prio'].sum()

        sample_inds = np.random.choice(self.capacity, n, p=prob)

        weights = (self.size * prob[sample_inds]) ** (-self.beta)
        weights /= weights.max()

        samples = {
            'inds': sample_inds,
            'weights': weights
        }
        for k, v in self.data.items():
            samples[k] = v[sample_inds]

        return samples

    def update_prio(self, ind: T.Tensor, prio: np.array):
        self.data['prio'][ind] = prio.clip(max=self.max_priority)

    @property
    def min_prio_idx(self):
        return self.data["prio"][self.data["prio"].nonzero()].argmin()

    @property
    def reward_sum(self):
        return (self.data['reward'] > 0).sum()

    @property
    def is_full(self):
        return self.capacity == self.size
