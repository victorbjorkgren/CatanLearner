from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch
from bidict import bidict
from torch import Tensor

from Environment.constants import N_NODES, N_ROADS, N_RESOURCES
from Learner.Utility.Utils import Holders


class BaseAction(Holders):
    pass


@dataclass
class TradeAction(BaseAction):
    give: Tensor
    get: Tensor


@dataclass
class BuildAction(BaseAction):
    mat_index: Tensor


@dataclass
class RoadAction(BaseAction):
    index: Tensor


@dataclass
class SettlementAction(BaseAction):
    index: Tensor


@dataclass
class NoopAction(BaseAction):
    pass


@dataclass
class Pi:
    type: Tensor
    map: Tensor
    trade: TradeAction


# dense_type_mapping = bidict({
#     0: NoopAction,
#     1: TradeAction,
#     2: BuildAction,
# })


@dataclass
class SparsePi(Holders):
    type: Tensor
    settlement: Tensor
    road: Tensor
    trade: TradeAction


sparse_type_mapping = bidict({
    0: NoopAction,
    1: TradeAction,
    2: SettlementAction,
    3: RoadAction,
})


@dataclass
class FlatPi(Holders):
    index: Tensor

    def get_noop_p(self) -> Tensor:
        noop_i = self.action_to_i(NoopAction())
        noop_p = self.index[:, :, noop_i, :]
        return noop_p

    def unstack_parts(self):
        # Assuming that the tensor dimensions match the expected ones from stack_parts
        # Extract dimensions
        b, t, total_actions, _ = self.index.shape

        # Calculate the specific sizes
        settle_size = N_NODES
        road_size = N_ROADS * 2
        trade_size = N_RESOURCES ** 2
        noop_size = 1

        settle_i = settle_size
        road_i = settle_size + road_size
        trade_i = settle_size + road_size + trade_size
        noop_i = settle_size + road_size + trade_size + noop_size

        # Split the stacked tensor based on these sizes
        settle = self.index[:, :, :settle_i, :]
        road = self.index[:, :, settle_i:road_i, :]
        flat_cross_probs = self.index[:, :, road_i:trade_i, :]
        noop = self.index[:, :, trade_i:noop_i, :]

        # Reshape the flat_cross_probs back to the original trade dimensions
        flat_cross_probs = flat_cross_probs.view(b, t, N_RESOURCES, N_RESOURCES, 1)

        # Split the flattened cross-probabilities back into 'give' and 'get'
        trade_give = flat_cross_probs.sum(dim=-2).squeeze(-1)
        trade_get = flat_cross_probs.sum(dim=-3).squeeze(-1)

        # Reconstruct trade
        trade = torch.stack([trade_give, trade_get], dim=-1)

        return settle, road, trade, noop

    @staticmethod
    def i_to_action(sample):
        if sample < N_NODES:
            return SettlementAction(index=sample)
        elif sample < (N_NODES + N_ROADS * 2):
            return RoadAction(index=sample - N_NODES)
        elif sample < (N_NODES + N_ROADS * 2 + N_RESOURCES ** 2):
            sample = sample - (N_NODES + N_ROADS * 2)
            give, get = divmod(sample.item(), N_RESOURCES)
            assert give < 5 and get < 5
            return TradeAction(give=torch.tensor([give]), get=torch.tensor([get]))
        elif sample == (N_NODES + N_ROADS * 2 + N_RESOURCES ** 2):
            return NoopAction()
        else:
            raise IndexError

    @staticmethod
    def action_to_i(action):
        if isinstance(action, SettlementAction):
            return action.index
        elif isinstance(action, RoadAction):
            return action.index + N_NODES
        elif isinstance(action, TradeAction):
            give = action.give.item()
            get = action.get.item()
            sample = (N_NODES + N_ROADS * 2) + give * N_RESOURCES + get
            return sample
        elif isinstance(action, NoopAction):
            return (N_NODES + N_ROADS * 2 + N_RESOURCES ** 2)
        else:
            raise ValueError("Unknown action type")

    @classmethod
    def stack_parts(cls, settle, road, trade, noop):
        assert trade.give.ndim == 4
        trade.give = trade.give.squeeze(-1)
        trade.get = trade.get.squeeze(-1)
        b, t, n = trade.give.shape
        cross_probs = trade.give.unsqueeze(-1) * trade.get.unsqueeze(-2)
        flat_cross_probs = cross_probs.reshape(b, t, n * n, 1)  # Corrected dimensions
        noop = noop.unsqueeze(-1)

        stack = torch.cat([settle, road, flat_cross_probs, noop], dim=-2)
        return cls(stack)

    @staticmethod
    def unflatten_trade(flat_index):
        giving_index = flat_index // N_RESOURCES
        getting_index = flat_index % N_RESOURCES
        return giving_index, getting_index