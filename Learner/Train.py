from typing import List, Tuple

import numpy as np
import torch as T
import torch.nn.functional as F
import torch.optim as optim

from Learner.Agents import BaseAgent
from Learner.PrioReplayBuffer import PrioReplayBuffer
from Learner.Nets import GameNet


class Trainer():
    def __init__(self,
                 q_net: GameNet,
                 target_net: GameNet,
                 batch_size: int,
                 dry_run: int,
                 reward_min: int,
                 gamma: float
                 ):
        self.q_net = q_net
        self.target_net = target_net
        self.batch_size = batch_size
        self.dry_run = dry_run
        self.reward_min = reward_min
        self.gamma = gamma
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4, weight_decay=1e-5)
        self.batch_range = T.arange(self.batch_size, dtype=T.long)

        self.available_agents = None
        self.tick = 0

        self.known_allowed_states = None

    def train(self) -> Tuple[float, float]:
        assert self.available_agents is not None, 'Trainer: No available agents'

        self.tick += 1
        if self.tick < self.dry_run:
            return 0., 0.

        agent = self.choose_agent()

        if agent.mem_size < self.batch_size:
            return 0., 0.
        else:
            sample = agent.sample(self.batch_size)

        inds = T.tensor(sample["inds"])
        next_inds = (inds + 1) % agent.mem_size
        weights = T.tensor(sample["weights"]).to(self.q_net.on_device)

        # Make sure state pairs belong to same episode, otherwise drop inds
        episode = agent.data['episode'][inds]  # .to(self.q_net.on_device)
        episode_next = agent.data['episode'][next_inds]  # .to(self.q_net.on_device)
        reward = agent.data['reward'][inds]
        done = agent.data['done'][inds].bool()
        inds = inds[(episode == episode_next) | (reward != 0) | done]
        next_inds = next_inds[(episode == episode_next) | (reward != 0) | done]
        weights = weights[(episode == episode_next) | (reward != 0) | done]
        batch_range = self.batch_range[:inds.shape[0]]

        state = agent.data['state'][inds].to(self.q_net.on_device)
        state_mask = agent.data['state_mask'][inds].to(self.q_net.on_device)
        new_state = agent.data['state'][next_inds].to(self.q_net.on_device)
        action = agent.data['action'][inds].to(self.q_net.on_device)
        reward = agent.data['reward'][inds].to(self.q_net.on_device)
        done = agent.data['done'][inds].bool().to(self.q_net.on_device)
        player = agent.data['player'][inds].to(self.q_net.on_device)

        # TODO: Check Norm of samples
        q = self.q_net(state)[batch_range, :, :, player]
        rule_breaking_q = self.get_rule_break_q(q.detach(), state_mask)
        q = q[batch_range, action[:, 0], action[:, 1]]

        # Get next action
        next_q = self.q_net(new_state)[batch_range, :, :, player]
        next_act, _ = self.get_batch_max(next_q)

        # Get next Q from target net
        with T.no_grad():
            next_q = self.target_net(new_state).detach()
            next_q[done] = 0
            next_q = next_q[batch_range, next_act[:, 0], next_act[:, 1], player]

        target_q = reward + (self.gamma * next_q)

        # Calculate loss
        td_error = F.mse_loss(q, target_q, reduction="none")
        td_error = td_error.clamp_max(2.)

        rule_break_loss = T.mean(rule_breaking_q ** 2)
        rule_break_loss = rule_break_loss.clamp_max(1.)

        agent.update_prio(inds, td_error.detach().cpu().numpy())
        loss = (td_error * weights).mean() + .001 * rule_break_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.tick % 100 == 0:
            self.target_net.clone_state(self.q_net)

        return td_error.mean().item(), rule_break_loss.item()

    def register_agents(self, agents: List[BaseAgent]):
        self.available_agents = agents

    def choose_agent(self) -> BaseAgent:
        assert self.available_agents is not None, 'Trainer: No available agents'
        """Chooses an agent based on the total prio for each agent."""
        prios = np.empty(len(self.available_agents))
        for i in range(len(self.available_agents)):
            prios[i] = self.available_agents[i].priority_sum
        prios[prios < 1e-6] = 1e-6
        choice = np.random.choice(len(self.available_agents), p=prios / prios.sum(), size=(1,))
        return self.available_agents[choice.item()]

    def save(self):
        super().save()
        self.q_net.save()

    def update_known_allowed(self, state_mask):
        if self.known_allowed_states is None:
            self.known_allowed_states = state_mask.clone().detach().any(0).squeeze()
        else:
            self.known_allowed_states = state_mask.clone().detach().any(0).squeeze() | self.known_allowed_states

    @staticmethod
    def get_batch_max(batched_tensor: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        max_act = T.empty((batched_tensor.shape[0], 2), dtype=T.long)
        max_q = T.empty((batched_tensor.shape[0],)).to(batched_tensor.device)
        for i in range(batched_tensor.shape[0]):
            b_inds = T.argwhere(batched_tensor[i] == batched_tensor[i].max())
            if b_inds.numel() > 2:
                b_inds = b_inds[T.randint(0, b_inds.shape[0], (1,))]
            max_act[i, :] = b_inds
            max_q[i] = batched_tensor[i].max()
        return max_act, max_q

    def get_rule_break_q(self, q: T.Tensor, state_mask: T.Tensor) -> T.Tensor:
        self.update_known_allowed(state_mask)
        rule_mask = (~state_mask.squeeze() & self.known_allowed_states)

        _, max_q = self.get_batch_max(q)
        rule_breaking_q = q - self.gamma * max_q[:, None, None]
        rule_breaking_q = rule_breaking_q[rule_mask]
        if rule_breaking_q.numel() == 0:
            rule_breaking_q = T.zeros((1,))

        return rule_breaking_q
