import os
import pickle
import threading
from abc import abstractmethod
from collections import defaultdict
from queue import Queue
from time import sleep

import numpy as np
import torch as T


def assert_capacity(n):
    # Check that n is an integer greater than 0 and n & (n - 1) == 0
    assert isinstance(n, int) and n > 0 and n & (n - 1) == 0, "Value is not a power of 2"


class PrioReplayBuffer:
    def __init__(self,
                 capacity: int,
                 max_seq_len: int,
                 alpha: float,
                 beta: float,
                 save_interval: int = 10_000
                 ) -> None:
        assert_capacity(capacity)

        _capacity: int
        _alpha: float
        _beta: float
        _max_priority: float
        _size: int
        data: dict
        _next_idx: int

        self.prios = np.zeros((capacity,), dtype=float)

        # if self.load():
        #     return

        self._capacity = capacity
        self._alpha = alpha
        self._beta = beta
        self._max_seq_len = max_seq_len

        self._save_interval = save_interval
        self._save_countdown = save_interval

        self._max_priority = 1.
        self._sum_priority = 0.

        # self._buffer = {}
        # self._req_data_keys = set(self.data.keys())
        # self._req_data_keys.remove('prio')

        self._next_idx = 0
        self._size = 0

    @property
    def mem_size(self):
        return self._size

    @property
    def min_prio_idx(self):
        return self.prios[self.prios.nonzero()].argmin()

    # @property
    # def reward_sum(self):
    #     return (self.data['reward'] > 0).sum()

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

    @abstractmethod
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

        pass

        # self.save_test()

    @abstractmethod
    def sample(self, n):
        pass

    def update_prio(self, ind: T.Tensor, prio: np.array):
        clipped_prio = prio.clip(max=self._max_priority)
        self.prios[ind] = clipped_prio

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

    def _sample_indices(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        prob = self.prios[:self._size] / max(1, self.prios[:self._size].sum())
        try:
            sample_inds = np.random.choice(self._size, n, p=prob, replace=False)
        except ValueError:
            sample_inds = np.random.choice(self._size, n, replace=False)
        return sample_inds, prob[sample_inds]


class InMemBuffer(PrioReplayBuffer):
    def __init__(self,
                 capacity: int,
                 max_seq_len: int,
                 alpha: float,
                 beta: float,
                 save_interval: int = 10_000
                 ) -> None:
        super(InMemBuffer, self).__init__(capacity, max_seq_len, alpha, beta)

        # TODO: Fix Magic Numbers
        self.data = {}
        self.data['state'] = T.zeros((capacity, max_seq_len, 74, 74, 16), dtype=T.float)
        self.data['seq_len'] = T.zeros((capacity,), dtype=T.long)
        self.data['action'] = T.zeros((capacity, max_seq_len, 2), dtype=T.long)
        self.data['was_trade'] = T.zeros((capacity, max_seq_len), dtype=T.bool)
        self.data['reward'] = T.zeros((capacity, max_seq_len), dtype=T.float)
        self.data['lstm_state'] = T.zeros((capacity, max_seq_len, 32), dtype=T.float)
        self.data['lstm_cell'] = T.zeros((capacity, max_seq_len, 32), dtype=T.float)
        self.data['done'] = T.zeros((capacity,), dtype=T.bool)
        self.data['episode'] = T.zeros((capacity,), dtype=T.long)
        self.data['player'] = T.zeros((capacity,), dtype=T.long)

    def sample(self, n):
        sample_inds, prob = self._sample_indices(n)

        weights = (self._size * prob) ** (-self._beta)
        weights = weights.clip(0, 1)

        samples = {
            'inds': sample_inds,
            'weights': weights
        }
        for k, v in self.data.items():
            samples[k] = v[sample_inds]

        return samples

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


class OnDiskBuffer(PrioReplayBuffer):
    def __init__(self,
                 capacity: int,
                 max_seq_len: int,
                 alpha: float,
                 beta: float,
                 batch_size: int,
                 preload_size: int,
                 save_interval: int = 10_000
                 ) -> None:
        super(OnDiskBuffer, self).__init__(capacity, max_seq_len, alpha, beta)

        self._path = 'D:/ReplayBuffer'
        os.makedirs(self._path, exist_ok=True)

        self._batch_size = batch_size
        self._preload_size = preload_size

        self.preload_queue = Queue(maxsize=preload_size)
        self._size = len(os.listdir(self._path))
        self._next_idx = self._size % self._capacity
        self.prios[:self._size] = self._max_priority ** self._alpha

        self.lock = threading.Lock()

        # self._preload_data()
        self.preload_thread = threading.Thread(target=self._preload_data, daemon=True)
        self.preload_thread.start()

    def _sample_indices(self, n: int):
        with self.lock:
            return super()._sample_indices(n)

    def _save_transition(self, index, transition):
        file_path = os.path.join(self._path, f'data_{index}.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(transition, f)

    def _load_transition(self, index) -> dict:
        file_path = os.path.join(self._path, f'data_{index}.pkl')
        max_attempts = 5
        attempt = 0
        while attempt < max_attempts:
            try:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            except:
                sleep(0.2)  # Wait for 0.2 seconds before retrying
                attempt += 1
        raise Exception(f"Failed to load data from {file_path} after {max_attempts} attempts.")

    def _preload_data(self):
        while True:
            if self.preload_queue.full() | (self._size < self._preload_size):
                sleep(.1)
                continue

            indices, probs = self._sample_indices(self._batch_size)
            transitions = [self._load_transition(index) for index in indices]

            transitions_dict = defaultdict(list)
            for transition in transitions:
                for key, value in transition.items():
                    if isinstance(value, float | int):
                        value = T.tensor(value)
                    transitions_dict[key].append(value)

            stacked_transitions = {key: T.stack(values) for key, values in transitions_dict.items()}
            stacked_transitions['probs'] = T.tensor(probs)
            self.preload_queue.put(stacked_transitions)

    def sample(self, n):
        assert n <= self._preload_size, 'Cannot get more samples than preload size'

        while self.preload_queue.qsize() < n:
            sleep(.01)

        transitions = self.preload_queue.get()
        weights = (self._size * transitions['probs']) ** (-self._beta)
        weights = weights.clip(0, 1)
        transitions['weights'] = weights
        return transitions

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

        idx = self._next_idx
        self._next_idx = (idx + 1) % self._capacity

        state = state.squeeze()
        reward = reward.squeeze()
        was_trade = was_trade.squeeze()

        seq_len = len(state)
        pad_len = self._max_seq_len - seq_len
        pad_lstm = self._max_seq_len - len(lstm_state)

        data = {}
        data['inds'] = idx
        data['state'] = T.constant_pad_nd(state, (0, 0, 0, 0, 0, 0, 0, pad_len), 0.)
        data['action'] = T.constant_pad_nd(action, (0, 0, 0, pad_len), 0.)
        data['was_trade'] = T.constant_pad_nd(was_trade, (0, pad_len), 0.)
        data['reward'] = T.constant_pad_nd(reward, (0, pad_len), 0.)
        data['lstm_state'] = T.constant_pad_nd(lstm_state, (0, 0, 0, pad_lstm), 0.)
        data['lstm_cell'] = T.constant_pad_nd(lstm_cell, (0, 0, 0, pad_lstm), 0.)
        data['seq_len'] = seq_len
        data['done'] = done
        data['episode'] = episode
        data['player'] = player
        if data['state'].shape[0] != data['lstm_state'].shape[0]:
            print('WARNING: state')
        self.prios[idx] = min(td_error.item(), self._max_priority) ** self._alpha

        self._size = min(self._capacity, self._size + 1)

        self._save_transition(idx, data)
