import torch as T
import torch.nn.functional as F
import torch.optim as optim

from Learner.PrioReplayBuffer import PrioReplayBuffer
from Learner.Nets import GameNet


class Trainer(PrioReplayBuffer):
    def __init__(self,
                 q_net: GameNet,
                 target_net: GameNet,
                 batch_size: int,
                 dry_run: int,
                 reward_min: int,
                 gamma: float,
                 memory_size: int,
                 alpha: float,
                 beta: float,
                 ):
        super().__init__(memory_size, alpha, beta)
        self.q_net = q_net
        self.target_net = target_net
        self.batch_size = batch_size
        self.dry_run = dry_run
        self.reward_min = reward_min
        self.gamma = gamma
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4, weight_decay=1e-5)

    def train(self, tick, episode):
        if self._size < max(self.batch_size, self.dry_run):
            return 0
        if self.reward_sum < self.reward_min:
            return 0

        self.optimizer.zero_grad()
        batch_range = T.arange(self.batch_size, dtype=T.long)

        # Keys: states, actions, new_states, rewards, dones, prio, inds, weights
        sample = self.sample(self.batch_size)
        reward = sample["reward"][batch_range, sample["player"]].to(self.q_net.on_device)
        weights = T.tensor(sample["weights"]).to(self.q_net.on_device)
        # player_mask = sample["player"][:, None, None, None].expand(-1, 74, 74, 1)

        # TODO: Check Norm of samples
        q = self.q_net(sample["state"])
        q = q[
            batch_range,
            sample["action"][:, 0],
            sample["action"][:, 1],
            sample["player"]
        ]

        # Get next action
        temp_q = self.q_net(sample["new_state"])
        temp_q = temp_q[batch_range, :, :, sample["player"].long()]
        max_indices_flat = T.argmax(temp_q.view(self.batch_size, -1), dim=1)

        next_act_row = max_indices_flat // 74
        next_act_col = max_indices_flat % 74

        with T.no_grad():
            next_q = self.target_net(sample["new_state"]).detach()
            next_q[sample["done"].bool()] = 0
            next_q = next_q[
                batch_range,
                next_act_row,
                next_act_col,
                sample["player"]
            ]

        target_q = reward + (self.gamma * next_q)

        # Calculate and remember loss
        td_error = F.mse_loss(q, target_q, reduction="none")
        td_error = td_error.clamp_max(1.)
        self.update_prio(sample['inds'], td_error.detach().cpu().numpy())
        loss = (td_error * weights).mean()

        loss.backward()
        self.optimizer.step()

        if episode % 2 == 0:
            self.target_net.clone_state(self.q_net)

        return td_error.mean().item()

    def save(self):
        super().save()
        self.q_net.save()
