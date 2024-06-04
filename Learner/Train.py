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
                 gamma: float,
                 memory_size: int,
                 alpha: float,
                 beta: float
                 ):
        super().__init__(memory_size, alpha, beta)
        self.q_net = q_net
        self.target_net = target_net
        self.batch_size = batch_size
        self.gamma = gamma
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)

    def train(self, tick, episode):
        if self.size < self.batch_size:
            return 0

        self.optimizer.zero_grad()

        # Keys: states, actions, new_states, rewards, dones, prio, inds, weights
        sample = self.sample(self.batch_size)

        q = self.q_net(sample["state"]).gather(1, sample["action"]).squeeze(1)
        next_action = self.q_net(sample["new_state"]).argmax(-1)

        with T.no_grad():
            next_q = self.target_net(sample["new_state"]).detach()
            next_q[sample["done"]] = 0

        target_q = sample["reward"] + (self.gamma * next_q[next_action])

        # Calculate and remember loss
        loss = F.mse_loss(q, target_q, reduction="none")
        self.update_prio(sample['inds'], loss.detach())
        loss = (loss * sample["weights"]).mean()

        loss.backward()
        self.optimizer.step()

        if episode % 2 == 0:
            self.target_net.clone_state(self.q_net)

        return loss.item()
