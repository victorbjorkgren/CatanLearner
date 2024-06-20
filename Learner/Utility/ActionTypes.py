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

import torch
from dataclasses import dataclass
from torch import Tensor

@dataclass
class FlatPi(Holders):
    index: Tensor

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
            return (N_NODES + N_ROADS * 2 + N_RESOURCES ** 2) + 1
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