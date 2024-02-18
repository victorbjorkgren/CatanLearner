from collections import deque

def assert_capacity(n):
    # Check that n is an integer greater than 0 and n & (n - 1) == 0
    assert isinstance(n, int) and n > 0 and n & (n - 1) == 0, "Value is not a power of 2"


class PrioReplayBuffer:
    def __init__(self, capacity, alpha):
        assert_capacity(capacity)

        self.capacity = capacity
        self.alpha = alpha

        self.priority_sum = [0] * (2 * capacity)
        self.priority_min = [float('inf')] * (2 * capacity)

        self.max_priority = 1.

        self.data = {
            'state': deque(maxlen=capacity),
            'action': deque(maxlen=capacity),
            'new_state': deque(maxlen=capacity),
            'reward': deque(maxlen=capacity),
            'done': deque(maxlen=capacity)
        }

        self.next_idx = 0
        self.size = 0

    def add(self, state, action, new_state, reward, done):
        idx = self.next_idx

        self.data['state'].append(state)
        self.data['action'].append(action)
        self.data['new_state'].append(new_state)
        self.data['reward'].append(reward)
        self.data['done'].append(done)

        self.next_idx = (idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

        priority_alpha = self.max_priority ** self.alpha
