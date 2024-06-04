from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from bidict import bidict
from torch import Tensor

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
class NoopAction(BaseAction):
    pass


@dataclass
class Pi:
    type: Tensor
    map: Tensor
    trade: TradeAction


type_mapping = bidict({
    0: NoopAction,
    1: TradeAction,
    2: BuildAction,
})