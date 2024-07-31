import os
from abc import abstractmethod
from dataclasses import fields
from time import sleep
from typing import Dict, List

import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Learner.Utility.ActionTypes import SparsePi, TradeAction, FlatPi
from Learner.Utility.CustomDistributions import SparseCatanActionSampler, FlatCatanActionSampler
from Learner.Utility.DataTypes import PPOTransition, NetInput, NetOutput
from Learner.constants import PPO_ENTROPY_COEF, PPO_VALUE_COEF, SAVE_LATEST_NET_INTERVAL, SAVE_CHECKPOINT_NET_INTERVAL, \
    SAC_ENTROPY_ALPHA, SAC_TARGET_TAU
from Learner.Loss import Loss
from Learner.Nets import GameNet, PPONet
from Learner.PrioReplayBuffer import PrioReplayBuffer
from Learner.Utility.Utils import TensorUtils, LinearSchedule
from Learner.constants import LOSS_CLIP, GRAD_CLIP


class Trainer:
    def __init__(self, name: str, buffer: PrioReplayBuffer, batch_size: int, gamma: float, net: GameNet, net_misc: List[GameNet] = []):
        self.name = name
        self.buffer = buffer
        self.gamma = gamma
        self.batch_size = batch_size
        self.net = net
        self.net_misc = net_misc

        self.known_allowed_states = None
        self.tick_iter = 0

        self.batch_range = T.arange(self.batch_size, dtype=T.long)

        self.find_start_tick()

    @abstractmethod
    def train(self) -> (float, Dict):
        raise NotImplementedError

    def tick(self) -> (float, Dict):
        if self.buffer.mem_size < self.batch_size:
            sleep(1)
            return 0., {}

        td_loss, stats = self.train()

        if self.tick_iter % SAVE_LATEST_NET_INTERVAL == 0:
            self.save('latest')
        if self.tick_iter % SAVE_CHECKPOINT_NET_INTERVAL == 0:
            self.save('checkpoint')

        # Sometimes print weight info
        # if self.tick_iter % 50 == 1:
        #     print('Print time')
        #     for name, module in self.net.named_modules():
        #         print(name)
        #         if hasattr(module, 'weight') and module.weight is not None:
        #             weight_max = T.max(module.weight).item()
        #             if module.weight.grad is not None:
        #                 grad_max = T.max(module.weight.grad).item()
        #             else:
        #                 grad_max = 0.
        #             print(f"{name}: Max weight = {weight_max:.4e} - Max grad = {grad_max:.4e}")

        self.tick_iter += 1

        return td_loss, stats

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
            for misc in self.net_misc:
                misc.save('latest')
        else:
            self.net.save(self.tick_iter)
            for misc in self.net_misc:
                misc.save(self.tick_iter)

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
        diff = pi_logprobs - behavior_logprob.squeeze(-1)
        ratio = torch.exp(diff.clamp(min=-20, max=20))
        clipped_ratios = torch.clamp(
            ratio,
            1.0 - self.clip_epsilon(self.tick_iter),
            1.0 + self.clip_epsilon(self.tick_iter)
        )
        policy_loss = torch.min(ratio * advantages.detach(), clipped_ratios * advantages.detach())
        return policy_loss

    # def apply_masks(self, net_output: NetOutput, masks: SparsePi):
    #     for field in fields(net_output.pi):
    #         if field.name == 'trade':
    #             value = net_output.pi.trade.give
    #             value = value * masks.trade.give.unsqueeze(-1)
    #             value[(value == 0).all(-2).squeeze(-1)] = 1.
    #             value = value / value.sum(dim=-2, keepdim=True)
    #             value[value.isnan()] = 1 / value.shape[2]
    #             net_output.pi.trade.give = value
    #         else:
    #             value = getattr(net_output.pi, field.name)
    #             value = value * getattr(masks, field.name).unsqueeze(-1)
    #             value[(value == 0).all(-2).squeeze(-1)] = 1.
    #             value = value / value.sum(dim=-2, keepdim=True)
    #             value[value.isnan()] = 1 / value.shape[2]
    #             setattr(net_output.pi, field.name, value)

        # pi_type = net_output.pi.type * masks.type.unsqueeze(-1)
        # pi_road = net_output.pi.road * masks.road.unsqueeze(-1)
        # pi_settle = net_output.pi.settlement * masks.settlement.unsqueeze(-1)
        # pi_trade_give = net_output.pi.trade.give * masks.trade.give.unsqueeze(-1)
        #
        # pi_type = pi_type / pi_type.sum(-1, keepdim=True)
        # pi_road = pi_road / pi_road.sum(-1, keepdim=True)
        # pi_settle = pi_settle / pi_settle.sum(-1, keepdim=True)
        # pi_trade_give = pi_trade_give / pi_trade_give.sum(-1, keepdim=True)
        #
        # return SparsePi(type=pi_type, settlement=pi_settle, road=pi_road, trade=TradeAction(give=pi_trade_give, get=net_output.pi_trade.get))

    def train(self) -> (float, Dict):
        assert self.buffer.mem_size >= self.batch_size
        stats = {}

        sample = self.buffer.sample(self.batch_size)
        transition: PPOTransition = sample["transition"].to(self.net.on_device)

        done = transition.reward_pack.done.any(1)
        value = transition.action_pack.value
        i_am_player = transition.reward_pack.player
        reward = transition.reward_pack.reward * self.reward_scale
        masks = transition.action_pack.masks

        value = TensorUtils.signed_log(value)

        assert done.all().item(), 'Currently only handles terminated sequences'
        advantage, returns = TensorUtils.advantage_estimation(reward, value, done, transition.seq_lens, self.gamma)
        # advantage, returns = TensorUtils.advantage_estimation(reward, torch.zeros_like(value), done, transition.seq_lens, self.gamma)
        # advantage = TensorUtils.propagate_rewards(self.gamma, reward)
        # returns = advantage.clone()
        stats['advantage'] = advantage.clone().detach().cpu()
        stats['returns'] = returns.clone().detach().cpu()

        net_output = self.net(sample['transition'].as_net_input)
        # Mask, choose correct player and renormalize
        pi = net_output.pi.index * masks.index
        pi = torch.gather(pi, -1, i_am_player[:,:,None,None].expand(-1,-1,224,-1)).squeeze(-1)
        pi = pi / pi.sum(-1, keepdims=True).clamp_min(1e-9)
        pi = FlatPi(pi.unsqueeze(-1))

        # self.apply_masks(net_output, masks)
        # pi = self.net.get_pi(net_output, i_am_player)

        pi_dist = FlatCatanActionSampler(pi)

        # Policy Loss
        pi_logprob = pi_dist.log_prob(transition.action_pack.action)
        policy_loss = self.proximal_policy_loss(pi_logprob, transition.action_pack.log_prob, advantage)

        # Value Loss
        # value = net_output.state_value.squeeze(-1)
        value_loss = 0.5 * torch.square(returns - value)

        # Entropy Loss
        entropy_loss = pi_dist.entropy()

        # Set loss out of sequence to zero
        b, t = value.shape
        out_of_seq_mask = (torch.arange(t)[None, :] >= transition.seq_lens[:, None].cpu()).to(self.net.on_device)
        policy_loss = policy_loss * ~out_of_seq_mask
        entropy_loss = entropy_loss * ~out_of_seq_mask
        value_loss = value_loss * ~out_of_seq_mask

        policy_loss = torch.mean(policy_loss)
        entropy_loss = torch.mean(entropy_loss)
        value_loss = torch.mean(value_loss)

        stats['policy_loss'] = -policy_loss.detach().cpu().item()
        stats['clip_epsilon'] = self.clip_epsilon(self.tick_iter)
        stats['value_loss'] = value_loss.detach().cpu().item()
        stats['entropy_loss'] = -entropy_loss.detach().cpu().item()

        # Combine policy loss, value loss, entropy loss.
        # Negative sign to indicate we want to maximize the policy gradient objective function and entropy to encourage exploration
        loss = -(policy_loss + PPO_ENTROPY_COEF * entropy_loss) + PPO_VALUE_COEF * value_loss
        stats['loss'] = loss.detach().cpu().item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), stats

class SACTrainer(Trainer):
    def __init__(self,
                 net: PPONet,
                 buffer: PrioReplayBuffer,
                 batch_size: int,
                 gamma: float,
                 learning_rate: float,
                 reward_scale: float
                 ):
        super().__init__('SAC_Agent', buffer, batch_size, gamma, net)
        self._q1_net = PPONet(net.config, net.batch_size, net.undirected_faces, name='Q1', softmax=False)
        self._q2_net = PPONet(net.config, net.batch_size, net.undirected_faces, name='Q2', softmax=False)

        self._q1_target_net = PPONet(net.config, net.batch_size, net.undirected_faces, name='Q1_target', softmax=False)
        self._q2_target_net = PPONet(net.config, net.batch_size, net.undirected_faces, name='Q2_target', softmax=False)

        if net.config['load_state']:
            self._q1_net.load('latest')
            self._q2_net.load('latest')

        self._alpha = SAC_ENTROPY_ALPHA

        self.update_targets()

        self.reward_scale = reward_scale

        self._policy_optimizer = optim.AdamW(self.net.parameters(), lr=learning_rate, weight_decay=1e-5)
        self._q1_optimizer = optim.AdamW(self._q1_net.parameters(), lr=learning_rate, weight_decay=1e-5)
        self._q2_optimizer = optim.AdamW(self._q2_net.parameters(), lr=learning_rate, weight_decay=1e-5)

    @staticmethod
    def gather_actions(values, actions):
        assert values.dim() == 3, 'Expected Tensor [b, t, f]'
        assert actions.dim() == 2, 'Expected Tensor [b, t]'

        selected_values = torch.gather(values, 2, actions.unsqueeze(-1).long())

        # Shape [b, t, 1] -> [b, t]
        selected_values = selected_values.squeeze(-1)

        return selected_values

    def update_targets(self):
        self._q1_target_net.clone_state(self._q1_net)
        self._q2_target_net.clone_state(self._q2_net)

    def proper_output(self, net, transitions):
        i_am_player = transitions.reward_pack.player
        masks = transitions.action_pack.masks
        net_output = net(transitions.as_net_input)
        pi = net_output.pi.index * masks.index
        pi = torch.gather(pi, -1, i_am_player[:,:,None,None].expand(-1,-1,224,-1)).squeeze(-1)
        pi = pi / pi.sum(-1, keepdims=True).clamp_min(1e-9)
        pi = FlatPi(pi.unsqueeze(-1))
        return pi.index.squeeze(-1)

    def q_loss_func(self, transitions):
        b, t, _, _ = transitions.tensor_size

        rewards = transitions.reward_pack.reward.clone()

        q1 = self.proper_output(self._q1_net, transitions)
        q2 = self.proper_output(self._q2_net, transitions)

        # q [b, t, n, f] -> [b, t]
        q1 = self.gather_actions(q1, transitions.action_pack.action)
        q2 = self.gather_actions(q2, transitions.action_pack.action)

        with torch.no_grad():
            q1_t = self.proper_output(self._q1_target_net, transitions)
            q2_t = self.proper_output(self._q2_target_net, transitions)
            a_prob = self.proper_output(self.net, transitions)
            q1_t = TensorUtils.left_shift_tensor(q1_t, 1)
            q2_t = TensorUtils.left_shift_tensor(q2_t, 1)
            a_prob = TensorUtils.left_shift_tensor(a_prob, 1)

            a_log_prob = torch.log(torch.clamp_min(a_prob, 1e-20))
            entropy = a_prob * a_log_prob

            min_q_target = torch.min(q1_t, q2_t)
            target_q = a_prob * (min_q_target - self._alpha * entropy)

            y = rewards + self.gamma * target_q.sum(-1)

        q1_loss = F.mse_loss(y, q1, reduction='none')
        q2_loss = F.mse_loss(y, q2, reduction='none')

        out_of_seq_mask = (torch.arange(t)[None, :] >= transitions.seq_lens[:, None].cpu()).to(self.net.on_device)
        q1_loss = q1_loss * ~out_of_seq_mask
        q2_loss = q2_loss * ~out_of_seq_mask

        q1_loss = q1_loss.mean(-1)
        q2_loss = q2_loss.mean(-1)

        return q1_loss, q2_loss

    def policy_loss_func(self, transitions):
        b, t, _, _ = transitions.tensor_size

        with torch.no_grad():
            q1 = self.proper_output(self._q1_net, transitions)
            q2 = self.proper_output(self._q2_net, transitions)

        q1 = TensorUtils.left_shift_tensor(q1, 1)
        q2 = TensorUtils.left_shift_tensor(q2, 1)

        a_probs = self.proper_output(self.net, transitions)
        a_probs = a_probs * transitions.action_pack.masks.index.squeeze(3)
        a_probs = a_probs / a_probs.sum(2, keepdim=True).clamp_min(1e-9)

        a_log_probs = torch.log(torch.clamp_min(a_probs, 1e-20))
        entropy = -torch.sum(a_probs * a_log_probs, dim=-1)

        min_q = a_probs * torch.min(q1, q2)
        min_q = torch.sum(min_q, dim=-1)

        # TODO: Implement learnable ent_coef
        # target_entropy = -torch.log(1.0 / transitions.state.action_mask.sum()) * 0.98
        # ent_coef_losses = self._log_ent_coef * (a_log_probs + target_entropy).detach()
        # ent_coef_loss = -torch.mean(torch.sum(ent_coef_losses, dim=-1), dim=0)

        policy_losses = -(min_q + self._alpha * entropy)

        loss_mask = (torch.arange(t)[None, :] >= transitions.seq_lens[:, None].cpu()).to(self.net.on_device)
        policy_losses = policy_losses * ~loss_mask

        policy_losses = policy_losses.view(b, -1).sum(-1)

        return policy_losses

    def train(self):
        assert self.buffer.mem_size >= self.batch_size
        stats = {}

        sample = self.buffer.sample(self.batch_size)
        transition: PPOTransition = sample["transition"].to(self.net.on_device)

        q1_loss, q2_loss = self.q_loss_func(transition)

        q1_loss = q1_loss.mean()
        q2_loss = q2_loss.mean()

        self._q1_optimizer.zero_grad()
        q1_loss.backward()
        self._q1_optimizer.step()

        self._q2_optimizer.zero_grad()
        q2_loss.backward()
        self._q2_optimizer.step()

        policy_loss = self.policy_loss_func(transition)

        policy_loss = policy_loss.mean()

        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        self._q1_target_net.lerp_towards(self._q1_net, SAC_TARGET_TAU)
        self._q2_target_net.lerp_towards(self._q2_net, SAC_TARGET_TAU)

        total_loss = q1_loss.detach() + q2_loss.detach() + policy_loss.detach()
        total_loss.cpu()

        stats['q1 loss'] = q1_loss.detach().cpu().item()
        stats['q2 loss'] = q2_loss.detach().cpu().item()
        stats['policy loss'] = policy_loss.detach().cpu().item()

        return total_loss, stats
