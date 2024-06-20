# Tests
import pytest
import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

from Learner.Utility.ActionTypes import FlatPi, SettlementAction, RoadAction, TradeAction, NoopAction


def test_i_to_action():
    assert FlatPi.i_to_action(5) == SettlementAction(index=5)
    assert FlatPi.i_to_action(12) == RoadAction(index=2)

    trade_action = FlatPi.i_to_action(50)
    assert isinstance(trade_action, TradeAction)
    assert trade_action.give.item() == 2
    assert trade_action.get.item() == 0

    assert isinstance(FlatPi.i_to_action(100), NoopAction)


def test_action_to_i():
    assert FlatPi.action_to_i(SettlementAction(index=5)) == 5
    assert FlatPi.action_to_i(RoadAction(index=2)) == 12

    trade_action = TradeAction(give=torch.tensor([2]), get=torch.tensor([0]))
    assert FlatPi.action_to_i(trade_action) == 50

    assert FlatPi.action_to_i(NoopAction()) == 101


def test_stack_parts():
    settle = torch.rand(2, 3, 10)
    road = torch.rand(2, 3, 30)
    trade_give = torch.rand(2, 3, 5, 1)
    trade_get = torch.rand(2, 3, 5, 1)
    noop = torch.rand(2, 3, 1)

    flatpi = FlatPi.stack_parts(settle, road, trade_give, trade_get, noop)

    expected_shape = (2, 3, 10 + 30 + 25 + 1)
    assert flatpi.index.shape == expected_shape


def test_unflatten_trade():
    give, get = FlatPi.unflatten_trade(10)
    assert give == 2
    assert get == 0


# Running the tests
if __name__ == "__main__":
    pytest.main([__file__])
