from time import sleep
from typing import Tuple

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Learner.Nets import GameNet
from Learner.PrioReplayBuffer import PrioReplayBuffer


class Trainer:
    def __init__(self,
                 q_net: GameNet,
                 target_net: GameNet,
                 buffer: PrioReplayBuffer,
                 batch_size: int,
                 gamma: float
                 ):
        self.q_net = q_net
        self.target_net = target_net
        self.buffer = buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4, weight_decay=1e-5)
        self.batch_range = T.arange(self.batch_size, dtype=T.long)

        self.tick_iter = 0

        self.known_allowed_states = None

    def tick(self):
        if self.buffer.mem_size < self.batch_size:
            sleep(1)
            return 0.

        td_loss = self.train()

        if self.tick_iter % 100 == 0:
            self.save('latest')
        if self.tick_iter % 200 == 0:
            self.save('checkpoint')

        self.tick_iter += 1

        return td_loss

    def train(self) -> float:
        assert self.buffer.mem_size >= self.batch_size

        sample = self.buffer.sample(self.batch_size)

        inds = T.tensor(sample["inds"])
        weights = T.tensor(sample["weights"]).to(self.q_net.on_device)
        batch_range = self.batch_range[:inds.shape[0]]

        state = self.buffer.data['state'][inds].to(self.q_net.on_device)
        action = self.buffer.data['action'][inds].to(self.q_net.on_device)
        reward = self.buffer.data['reward'][inds].to(self.q_net.on_device)
        done = self.buffer.data['done'][inds].bool().to(self.q_net.on_device)
        player = self.buffer.data['player'][inds].to(self.q_net.on_device)
        seq_lens = self.buffer.data['seq_len'][inds].long()
        lstm_state = self.buffer.data['lstm_state'][None, inds, 0, :].to(self.q_net.on_device)
        lstm_cell = self.buffer.data['lstm_cell'][None, inds, 0, :].to(self.q_net.on_device)

        lstm_target_state = self.buffer.data['lstm_state'][None, inds, -1, :].to(self.q_net.on_device)
        lstm_cell_state = self.buffer.data['lstm_cell'][None, inds, -1, :].to(self.q_net.on_device)

        q, _, _ = self.q_net(state, seq_lens, lstm_state, lstm_cell)
        q = q[batch_range, :, :, :, player]

        next_act, _ = self.get_batch_max(q[:, -1:])
        q = self.gather_actions(q, action)

        td_error = self.get_td_error(
            q,
            state,
            reward,
            next_act,
            seq_lens,
            lstm_target_state,
            lstm_cell_state,
            done,
            player
        )

        loss = (td_error * weights[:, None]).mean()  # + .001 * rule_break_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), .5)
        self.optimizer.step()

        # Sometimes print weight info
        if self.tick_iter % 1000 == 0:
            for name, module in self.q_net.named_modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    weight_max = T.max(module.weight).item()
                    grad_max = T.max(module.weight.grad).item()
                    print(f"{name}: Max weight = {weight_max:.4e} - Max grad = {grad_max:.4e}")

        # Update others
        if self.tick_iter % 100 == 0:
            self.target_net.clone_state(self.q_net)

        self.buffer.update_prio(inds, td_error.mean(-1).detach().cpu().numpy())

        return td_error.mean().item()

    def get_td_error(
            self,
            q: T.Tensor,
            state: T.Tensor,
            reward: T.Tensor,
            next_act: T.Tensor,
            seq_lens: T.Tensor,
            lstm_target_state: T.Tensor,
            lstm_cell_state: T.Tensor,
            done: T.Tensor,
            player: T.Tensor
    ):
        # Get next Q from target net
        with T.no_grad():
            next_q, _, _ = self.target_net(
                state[:, -1:, :, :],
                T.ones_like(seq_lens),
                lstm_target_state,
                lstm_cell_state
            )
            next_q = next_q[self.batch_range[q.shape[0]], :, :, :, player]
            next_q = self.gather_actions(next_q, next_act)

        reward[~done, seq_lens[~done.cpu()] - 1] = next_q[~done, 0]
        reward = self.propagate_rewards(self.gamma, reward)
        target_q = reward[:, :-1]

        for i in range(q.shape[0]):
            if seq_lens[i] == q.shape[1]:
                continue
            q[i, seq_lens[i]:] = 0
            target_q[i, seq_lens[i]:] = 0

        q = q[:, :-1]

        # Calculate loss
        td_error = F.mse_loss(q, target_q, reduction="none")

        return td_error
    
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

    def save(self, method):
        assert method in ['latest', 'checkpoint']
        if method == 'latest':
            self.q_net.save('latest')
        else:
            self.q_net.save(self.tick_iter)

    def update_known_allowed(self, state_mask):
        if self.known_allowed_states is None:
            self.known_allowed_states = state_mask.clone().detach().any(0).squeeze()
        else:
            self.known_allowed_states = state_mask.clone().detach().any(0).squeeze() | self.known_allowed_states

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

    @staticmethod
    def gather_actions(values, indices):
        return values[
            T.arange(values.size(0)).unsqueeze(1),
            T.arange(values.size(1)),
            indices[:, :, 0],
            indices[:, :, 1]
        ]

    def get_rule_break_q(self, q: T.Tensor, state_mask: T.Tensor) -> T.Tensor:
        self.update_known_allowed(state_mask)
        rule_mask = (~state_mask.squeeze() & self.known_allowed_states)

        _, max_q = self.get_batch_max(q)
        rule_breaking_q = q - self.gamma * max_q[:, None, None]
        rule_breaking_q = rule_breaking_q[rule_mask]
        if rule_breaking_q.numel() == 0:
            rule_breaking_q = T.zeros((1,))

        return rule_breaking_q
