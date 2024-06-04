from typing import Optional

import torch as T
import torch.nn.functional as F

from Learner.Nets import GameNet
from Learner.Utils import TensorUtils


class Loss:
    @staticmethod
    def get_td_error(
            target_net: GameNet,
            q: T.Tensor,
            next_state: T.Tensor,
            reward: T.Tensor,
            gamma: float,
            next_act: Optional[T.Tensor],
            seq_lens: T.Tensor,
            lstm_target_state: T.Tensor,
            lstm_cell_state: T.Tensor,
            done: T.Tensor,
            player: T.Tensor
    ):
        """
        NOTE: Assumes next action to be a build or pass action due to simplicity
        :param target_net:
        :param q:
        :param next_state:
        :param reward:
        :param gamma:
        :param next_act:
        :param seq_lens:
        :param lstm_target_state:
        :param lstm_cell_state:
        :param done:
        :param player:
        :return:
        """
        assert (next_act is not None) | (next_state.shape[0] == 1), "Passing None as next_act only available for batch size 1"

        batch_range = T.arange(next_state.shape[0], dtype=T.long)

        # Get next Q from target net
        with T.no_grad():
            next_q, trade_q, _, _ = target_net(
                next_state,
                T.ones_like(seq_lens),
                lstm_target_state,
                lstm_cell_state
            )
            next_q = next_q[batch_range, :, :, :, player]
            if next_act is None:
                next_q = next_q.max().view((1, -1))
            else:
                next_q = TensorUtils.gather_actions(next_q, trade_q, next_act, None)

        reward[~done, seq_lens[~done.cpu()] - 1] = next_q[~done, 0]
        reward = TensorUtils.propagate_rewards(gamma, reward)
        target_q = reward[:, :-1]

        for i in range(q.shape[0]):
            if seq_lens[i] == q.shape[1]:
                continue
            q[i, seq_lens[i]:] = 0
            target_q[i, seq_lens[i]:] = 0

        q = q[:, :-1]

        # Calculate loss
        try:
            td_error = F.mse_loss(q, target_q, reduction="none")
        except:
            print("Exception in td_error = F.mse_loss(q, target_q, reduction='none')")

        return td_error
