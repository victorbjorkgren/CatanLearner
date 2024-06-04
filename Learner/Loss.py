from typing import Tuple

import torch as T
import torch.nn as nn


class QFuncLoss(nn.Module):
    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = gamma
        self.huber_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self,
                q: T.Tensor,
                action: T.Tensor,
                double_q: T.Tensor,
                target_q: T.Tensor,
                done: T.Tensor,
                reward: T.Tensor,
                weights: T.Tensor
                ) -> Tuple[T.Tensor, T.Tensor]:
        q_sampled_action = q.gather(-1, action.to(T.long).unsqueeze(-1)).squeeze(-1)
        with T.no_grad():
            best_next_action = T.argmax(double_q, -1)
            best_next_q_value = target_q.gather(-1, best_next_action.unsqueeze(-1)).squeeze(-1)

            q_update = reward + self.gamma * best_next_q_value * (1 - done)
            td_error = q_sampled_action - q_update

        loss = self.huber_loss(q_sampled_action, q_update)
        loss = T.mean(weights * loss)

        return loss, td_error
