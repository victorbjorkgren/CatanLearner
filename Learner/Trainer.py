import os
from abc import abstractmethod
from time import sleep

import torch
import torch as T
import torch.nn as nn
import torch.optim as optim

from Learner.Utility.DataTypes import PPOTransition, NetInput
from Learner.constants import PPO_ENTROPY_COEF, PPO_VALUE_COEF, SAVE_LATEST_NET_INTERVAL, SAVE_CHECKPOINT_NET_INTERVAL
from Learner.Utility.CustomDistributions import CatanActionSampler
from Learner.Loss import Loss
from Learner.Nets import GameNet, PPONet
from Learner.PrioReplayBuffer import PrioReplayBuffer
from Learner.Utility.Utils import TensorUtils, LinearSchedule
from Learner.constants import LOSS_CLIP, GRAD_CLIP


class Trainer:
    def __init__(self, name: str, buffer: PrioReplayBuffer, batch_size: int, gamma: float, net: GameNet):
        self.name = name
        self.buffer = buffer
        self.gamma = gamma
        self.batch_size = batch_size
        self.net = net

        self.known_allowed_states = None
        self.tick_iter = 0

        self.batch_range = T.arange(self.batch_size, dtype=T.long)

        self.find_start_tick()

    @abstractmethod
    def train(self) -> float:
        raise NotImplementedError

    def tick(self):
        if self.buffer.mem_size < self.batch_size:
            sleep(1)
            return 0.

        td_loss = self.train()

        if self.tick_iter % SAVE_LATEST_NET_INTERVAL == 0:
            self.save('latest')
        if self.tick_iter % SAVE_CHECKPOINT_NET_INTERVAL == 0:
            self.save('checkpoint')

        self.tick_iter += 1

        return td_loss

    def find_start_tick(self):
        str_start = len(self.name) + 1
        str_end = len(".pth")
        os.makedirs('./PastTitans/', exist_ok=True)
        files = os.listdir('./PastTitans/')
        checkpoints = [int(f[str_start:-str_end]) for f in files] + [-1]
        self.tick_iter = 1 + max(checkpoints)

    def save(self, method):
        assert method in ['latest', 'checkpoint']
        if method == 'latest':
            self.net.save('latest')
        else:
            self.net.save(self.tick_iter)

    def update_known_allowed(self, state_mask):
        if self.known_allowed_states is None:
            self.known_allowed_states = state_mask.clone().detach().any(0).squeeze()
        else:
            self.known_allowed_states = state_mask.clone().detach().any(0).squeeze() | self.known_allowed_states


class QTrainer(Trainer):
    def __init__(self,
                 q_net: GameNet,
                 target_net: GameNet,
                 buffer: PrioReplayBuffer,
                 batch_size: int,
                 gamma: float,
                 learning_rate: float,
                 reward_scale: float
                 ):
        super().__init__(buffer, batch_size, gamma, q_net)
        self.name = 'Q_Agent'
        self.q_net = q_net
        self.target_net = target_net
        self.reward_scale = reward_scale
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate, weight_decay=1e-5)

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

    def get_rule_break_q(self, q: T.Tensor, state_mask: T.Tensor) -> T.Tensor:
        self.update_known_allowed(state_mask)
        rule_mask = (~state_mask.squeeze() & self.known_allowed_states)

        _, max_q = TensorUtils.get_batch_max(q)
        rule_breaking_q = q - self.gamma * max_q[:, None, None]
        rule_breaking_q = rule_breaking_q[rule_mask]
        if rule_breaking_q.numel() == 0:
            rule_breaking_q = T.zeros((1,))

        return rule_breaking_q


class PPOTrainer(Trainer):
    def __init__(self,
                 net: PPONet,
                 buffer: PrioReplayBuffer,
                 batch_size: int,
                 gamma: float,
                 learning_rate: float,
                 reward_scale: float
                 ):
        super().__init__('PPO_Agent', buffer, batch_size, gamma, net)
        self.reward_scale = reward_scale
        self.optimizer = optim.AdamW(self.net.parameters(), lr=learning_rate, weight_decay=1e-5)

        self.clip_epsilon = LinearSchedule(
            begin_t=0,
            end_t=100_000_000,  # Learner step_t is often faster than worker TODO: Adapt this?
            begin_value=.12,
            end_value=.02,
        )

    def proximal_policy_loss(self, pi_logprobs, behavior_logprob, advantages):
        ratio = torch.exp(pi_logprobs.squeeze(-1) - behavior_logprob)
        clipped_ratios = torch.clamp(
            ratio,
            1.0 - self.clip_epsilon(self.tick_iter),
            1.0 + self.clip_epsilon(self.tick_iter)
        )
        policy_loss = torch.min(ratio * advantages.detach(), clipped_ratios * advantages.detach())
        return policy_loss

    def train(self) -> float:
        assert self.buffer.mem_size >= self.batch_size

        sample = self.buffer.sample(self.batch_size)
        inds = sample["inds"]
        transition: PPOTransition = sample["transition"].to(self.net.on_device)
        batch_range = self.batch_range[:inds.shape[0]]

        done = transition.reward_pack.done.any(1)
        value = transition.action_pack.value
        i_am_player = transition.reward_pack.player
        # if done.sum() > 0:
        #     breakpoint()
        b, t = value.shape
        mask = (torch.arange(t)[None, :] >= transition.seq_lens.squeeze()[:, None]).to(self.net.on_device)
        mask = TensorUtils.pop_elements(mask, done)

        reward = transition.reward_pack.reward * self.reward_scale
        advantage, returns = TensorUtils.advantage_estimation(reward, value, done, transition.seq_lens, self.gamma)

        net_output = self.net(sample['transition'].as_net_input)
        pi = self.net.get_pi(net_output, i_am_player)

        pi_dist = CatanActionSampler(pi)

        # Policy Loss
        pi_logprob = pi_dist.log_prob(transition.action_pack.action)
        pi_logprob = TensorUtils.pop_elements(pi_logprob, done)
        pi_act_logprob = TensorUtils.pop_elements(transition.action_pack.log_prob, done)
        policy_loss = self.proximal_policy_loss(pi_logprob, pi_act_logprob, advantage)

        # Value Loss
        value = TensorUtils.pop_elements(value, done)
        value_loss = 0.5 * torch.square(returns - value)

        # Entropy Loss
        entropy_loss = pi_dist.entropy()
        entropy_loss = TensorUtils.pop_elements(entropy_loss, done)

        # Set loss out of sequence to zero
        policy_loss = policy_loss * mask
        entropy_loss = entropy_loss * mask
        value_loss = value_loss * mask

        policy_loss = torch.mean(policy_loss)
        entropy_loss = torch.mean(entropy_loss)
        value_loss = torch.mean(value_loss)

        # Combine policy loss, value loss, entropy loss.
        # Negative sign to indicate we want to maximize the policy gradient objective function and entropy to encourage exploration
        loss = -(policy_loss + PPO_ENTROPY_COEF * entropy_loss) + PPO_VALUE_COEF * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
