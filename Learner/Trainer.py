import os
from time import sleep

import torch as T
import torch.nn as nn
import torch.optim as optim

from Learner.Loss import Loss
from Learner.Nets import GameNet
from Learner.PrioReplayBuffer import PrioReplayBuffer
from Learner.Utils import TensorUtils
from Learner.constants import LOSS_CLIP, GRAD_CLIP


class Trainer:
    def __init__(self,
                 q_net: GameNet,
                 target_net: GameNet,
                 buffer: PrioReplayBuffer,
                 batch_size: int,
                 gamma: float,
                 learning_rate: float,
                 reward_scale: float
                 ):
        self.q_net = q_net
        self.target_net = target_net
        self.buffer = buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.batch_range = T.arange(self.batch_size, dtype=T.long)

        self.tick_iter = 0
        self.find_start_tick()

        self.known_allowed_states = None

    def find_start_tick(self):
        str_start = len("Q_Agent") + 1
        str_end = len(".pth")
        os.makedirs('./PastTitans/', exist_ok=True)
        files = os.listdir('./PastTitans/')
        checkpoints = [int(f[str_start:-str_end]) for f in files] + [0]
        self.tick_iter = 1 + max(checkpoints)

    def tick(self):
        if self.buffer.mem_size < self.batch_size:
            sleep(1)
            return 0.

        td_loss = self.train()

        if self.tick_iter % 100 == 0:
            self.save('latest')
        if self.tick_iter % 2000 == 0:
            self.save('checkpoint')

        self.tick_iter += 1

        return td_loss

    def train(self) -> float:
        assert self.buffer.mem_size >= self.batch_size

        sample = self.buffer.sample(self.batch_size)

        inds = sample["inds"]
        weights = sample["weights"].to(self.q_net.on_device)
        batch_range = self.batch_range[:inds.shape[0]]

        state = sample['state'].to(self.q_net.on_device)
        action = sample['action'].to(self.q_net.on_device)
        was_trade = sample['was_trade'].to(self.q_net.on_device)
        reward = sample['reward'].to(self.q_net.on_device)
        done = sample['done'].bool().to(self.q_net.on_device)
        player = sample['player'].to(self.q_net.on_device)
        seq_lens = sample['seq_len'].long()
        lstm_state = sample['lstm_state'][:, 0, :].unsqueeze(0).to(self.q_net.on_device)
        lstm_cell = sample['lstm_cell'][:, 0, :].unsqueeze(0).to(self.q_net.on_device)

        lstm_target_state = sample['lstm_state'][batch_range, seq_lens-2, :].unsqueeze(0).to(self.q_net.on_device)
        lstm_target_cell = sample['lstm_cell'][batch_range, seq_lens-2, :].unsqueeze(0).to(self.q_net.on_device)
        target_state = state[batch_range, seq_lens-2].unsqueeze(1)

        reward = reward * self.reward_scale

        q, trade_q, _, _ = self.q_net(state, seq_lens, lstm_state, lstm_cell)
        q = q[batch_range, :, :, :, player]
        trade_q = trade_q[batch_range, :, :, :, player]

        next_q, _, _, _ = self.q_net(target_state, T.tensor([1]), lstm_target_state, lstm_target_cell)
        next_act, _ = TensorUtils.get_batch_max(next_q[batch_range, :, :, :, player])
        q = TensorUtils.gather_actions(q, trade_q, action, was_trade)

        td_error = Loss.get_td_error(
            self.target_net,
            q,
            target_state,
            reward,
            self.gamma,
            next_act,
            seq_lens,
            lstm_target_state,
            lstm_target_cell,
            done,
            player
        )

        td_error = T.clamp(td_error, 0., LOSS_CLIP)
        loss = (td_error * weights[:, None]).mean()  # + .001 * rule_break_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

        # Sometimes print weight info
        if self.tick_iter % 1000 == 0:
            for name, module in self.q_net.named_modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    weight_max = T.max(module.weight).item()
                    if module.weight.grad is not None:
                        grad_max = T.max(module.weight.grad).item()
                    else:
                        grad_max = 0.
                    print(f"{name}: Max weight = {weight_max:.4e} - Max grad = {grad_max:.4e}")

        # Update others
        if self.tick_iter % 100 == 0:
            self.target_net.clone_state(self.q_net)

        self.buffer.update_prio(inds, td_error.mean(-1).detach().cpu().numpy())

        return td_error.mean().item()

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

    def get_rule_break_q(self, q: T.Tensor, state_mask: T.Tensor) -> T.Tensor:
        self.update_known_allowed(state_mask)
        rule_mask = (~state_mask.squeeze() & self.known_allowed_states)

        _, max_q = TensorUtils.get_batch_max(q)
        rule_breaking_q = q - self.gamma * max_q[:, None, None]
        rule_breaking_q = rule_breaking_q[rule_mask]
        if rule_breaking_q.numel() == 0:
            rule_breaking_q = T.zeros((1,))

        return rule_breaking_q
