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

    def train(self) -> Tuple[float, float]:
        assert self.available_agents is not None, 'Trainer: No available agents'

        self.tick += 1
        if self.tick < self.dry_run:
            return 0., 0.

        agent = self.choose_agent()

        if agent.mem_size < self.batch_size:
            return 0., 0.

        sample = agent.sample(self.batch_size)
        weights = T.tensor(sample["weights"]).to(self.q_net.on_device)
        state = agent.data['state'][sample["inds"]].to(self.q_net.on_device)
        state_mask = agent.data['state_mask'][sample["inds"]].to(self.q_net.on_device)
        new_state = agent.data['state'][sample["inds"] + 1].to(self.q_net.on_device)
        action = agent.data['action'][sample["inds"]].to(self.q_net.on_device)
        reward = agent.data['reward'][sample["inds"]].to(self.q_net.on_device)
        done = agent.data['done'][sample["inds"]].bool().to(self.q_net.on_device)
        episode = agent.data['episode'][sample["inds"]].to(self.q_net.on_device)
        player = agent.data['player'][sample["inds"]].to(self.q_net.on_device)

        # TODO: Check Norm of samples
        q = self.q_net(state)
        rule_breaking_q = q[self.batch_range, :, :, player][~state_mask.squeeze()]
        q = q[self.batch_range, action[:, 0], action[:, 1], player]

        # Get next action
        next_q = self.q_net(new_state)
        next_q = next_q[self.batch_range, :, :, player]
        max_indices_flat = T.argmax(next_q.view(self.batch_size, -1), dim=1)
        next_act_row = max_indices_flat // 74
        next_act_col = max_indices_flat % 74

        # Get next Q from target net
        with T.no_grad():
            next_q = self.target_net(new_state).detach()
            next_q[done] = 0
            next_q = next_q[self.batch_range, next_act_row, next_act_col, player]

        target_q = reward + (self.gamma * next_q)

        # Calculate and remember loss
        td_error = F.mse_loss(q, target_q, reduction="none")
        td_error = td_error.clamp_max(1.)
        rule_break_loss = T.mean(rule_breaking_q ** 2)
        rule_break_loss = rule_break_loss.clamp_max(1.)
        agent.update_prio(sample['inds'], td_error.detach().cpu().numpy())
        loss = (td_error * weights).mean() + rule_break_loss

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
