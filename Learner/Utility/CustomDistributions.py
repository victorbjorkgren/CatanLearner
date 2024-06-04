import torch
from torch import Tensor
from torch.distributions import Categorical

from Learner.Utility.ActionTypes import BaseAction, TradeAction, BuildAction, NoopAction, type_mapping


class CatanActionSampler:
    def __init__(self, pi_type, pi_map, pi_trade):
        self.build_size = pi_map.shape[1]
        self.type_dist = Categorical(probs=pi_type)
        self.build_dist = Categorical(probs=pi_map.flatten())
        self.trade_give_dist = Categorical(probs=pi_trade.give)
        self.trade_get_dist = Categorical(probs=pi_trade.get)

    def sample(self):
        action_ind = self.type_dist.sample()
        action_type = type_mapping[action_ind.item()]

        if action_type == BuildAction:
            return BuildAction(mat_index=self.sample_2d_map())
        elif action_type == TradeAction:
            give = self.trade_give_dist.sample()
            get = self.trade_get_dist.sample()
            return TradeAction(give=give, get=get)
        elif action_type == NoopAction:
            return NoopAction()
        else:
            raise IndexError

    def log_prob(self, action: BaseAction):
        action_type = type(action)
        action_ind = type_mapping.inverse[action_type]

        type_log_p = self.type_dist.log_prob(torch.tensor(action_ind))
        if action_type == BuildAction:
            action: BuildAction
            action_log_p = self.logprob_2d_map(action.mat_index)
        elif action_type == TradeAction:
            action: TradeAction
            trade_give_log_p = self.trade_give_dist.log_prob(action.give)
            trade_get_log_p = self.trade_get_dist.log_prob(action.get)
            action_log_p = trade_give_log_p + trade_get_log_p
        elif action_type == NoopAction:
            action: NoopAction
            action_log_p = 0.

        return type_log_p + action_log_p

    def entropy(self):
        action_entropy = self.type_dist.entropy()
        action_probs = self.type_dist.probs

        build_probs = action_probs[..., 0]
        trade_probs = action_probs[..., 1]

        build_entropy = build_probs * self.build_dist.entropy()
        trade_entropy = trade_probs * self.trade_dist.entropy()

        return action_entropy + build_entropy + trade_entropy

    def sample_2d_map(self):
        sampled_index = self.build_dist.sample()
        sampled_coordinate = divmod(sampled_index.item(), self.build_size)
        return sampled_coordinate

    def logprob_2d_map(self, action: Tensor):
        row, col = action
        index = row * self.build_size + col
        return self.build_dist.log_prob(torch.tensor(index, dtype=torch.float, requires_grad=True))