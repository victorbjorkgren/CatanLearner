import torch
from torch import Tensor
from torch.distributions import Categorical

from Learner.Utility.ActionTypes import BaseAction, TradeAction, BuildAction, NoopAction, type_mapping, Pi


class CatanActionSampler:
    def __init__(self, pi: Pi):
        self.device = pi.map.device
        self.build_size = pi.map.shape[1]
        self.type_dist = Categorical(probs=pi.type)
        self.build_dist = Categorical(probs=pi.map.flatten())
        self.trade_give_dist = Categorical(probs=pi.trade.give)
        self.trade_get_dist = Categorical(probs=pi.trade.get)

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
        if isinstance(action, list):
            build_ind = type_mapping.inverse[BuildAction]
            trade_ind = type_mapping.inverse[TradeAction]
            noop_ind = type_mapping.inverse[NoopAction]

            b = len(action)
            t = max([len(e) for e in action])
            action_ind = torch.full((b, t), noop_ind, dtype=torch.float)
            build_mat_index = torch.zeros((b, t, 2), dtype=torch.float)
            trade_gives = torch.zeros((b, t), dtype=torch.float)
            trade_gets = torch.zeros((b, t), dtype=torch.float)
            for i in range(b):
                for j in range(t):
                    if j < len(action[i]):
                        action_type = type_mapping.inverse[type(action[i][j])]
                        action_ind[i, j] = action_type
                        if action_type == BuildAction:
                            build_mat_index[i, j, :] = action[i][j].mat_index
                        elif action_type == TradeAction:
                            trade_gives[i, j] = action[i][j].give
                            trade_gets[i, j] = action[i][j].get


            type_log_p = self.type_dist.log_prob(action_ind.to(self.device))
            build_log_p = self.logprob_2d_map(build_mat_index.to(self.device))
            trade_give_log_p = self.trade_give_dist.log_prob(trade_gives.to(self.device))
            trade_get_log_p = self.trade_get_dist.log_prob(trade_gets.to(self.device))

            builds = action_ind==build_ind
            trades = action_ind==trade_ind

            log_p = type_log_p
            log_p[builds] = log_p[builds] + build_log_p[builds]
            log_p[trades] = log_p[trades] + trade_give_log_p[trades] + trade_get_log_p[trades]

            return log_p
        else:
            action_type = type(action)
            action_ind = type_mapping.inverse[action_type]
            action_ind = torch.tensor(action_ind)

            type_log_p = self.type_dist.log_prob(action_ind)
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
            else:
                raise IndexError

            return type_log_p + action_log_p

    def entropy(self):
        action_entropy = self.type_dist.entropy()
        action_probs = self.type_dist.probs

        build_probs = action_probs[..., 0]
        trade_probs = action_probs[..., 1]

        get_entropy = self.trade_get_dist.entropy()
        give_entropy = self.trade_give_dist.entropy()
        trade_entropy = trade_probs * (get_entropy + give_entropy)
        build_entropy = build_probs * self.build_dist.entropy()

        return action_entropy + build_entropy + trade_entropy

    def sample_2d_map(self):
        sampled_index = self.build_dist.sample()
        sampled_coordinate = divmod(sampled_index.item(), self.build_size)
        return torch.tensor(sampled_coordinate)

    def logprob_2d_map(self, mat_index: Tensor):
        if mat_index.ndim == 1:
            row, col = mat_index
        else:
            row = mat_index[..., 0]
            col = mat_index[..., 1]
        index = row * self.build_size + col
        return self.build_dist.log_prob(index)