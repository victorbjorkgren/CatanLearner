from __future__ import annotations
from typing import Optional

import torch as T

from Learner.Agents import BaseAgent


###
# Player states:
#
# TODO: add cards etc.
# [Bricks, Grains, Ores, Lumbers, Wools]
#
class Player:
    def __init__(self, agent: BaseAgent | None):
        self.hand = T.tensor([0, 0, 0, 0, 0])
        self.points = 0
        self.n_settlements = 0
        self.n_cities = 0
        self.n_roads = 0
        self.agent = agent

        self.latent_reward = 0.

        self.trade_pile = T.tensor([0, 0, 0, 0, 0])
        self.best_trade_rate = T.tensor([4, 4, 4, 4, 4])

        # Add resources for first turn
        self.add(0, 4)
        self.add(1, 2)
        self.add(3, 4)
        self.add(4, 2)

    def __str__(self):
        # [Bricks, Grains, Ores, Lumbers, Wools]
        return ((f"Brick {int(self.hand[0])}\n"
                 f"Grain {int(self.hand[1])}\n"
                 f"Ore {int(self.hand[2])}\n"
                 f"Lumber {int(self.hand[3])}\n"
                 f"Wool {int(self.hand[4])}\n")
                + f"\nPoints {int(self.points)}")

    def flush_reward(self):
        r = self.latent_reward
        self.latent_reward = 0
        return r

    @property
    def state(self):
        return T.cat((self.hand, T.tensor([self.points])))

    def add(self, ind: int, n: int):
        self.hand[int(ind)] += int(n)

    def sub(self, ind: int, n: int):
        self.hand[ind] -= n

    def get(self, cards: T.Tensor):
        self.hand += cards

    def give(self, cards: T.Tensor, other: Optional[Player] = None):
        assert (self.hand >= cards).all(), "Player does not have those resources"
        self.hand -= cards
        if other is not None:
            other.get(cards)

    def trade(self, give_ind: T.Tensor, get_ind: T.Tensor) -> bool:
        give_ind = int(give_ind.item())
        get_ind = int(get_ind.item())

        if self.hand[give_ind] < 4:
            return False

        self.sub(give_ind, 4)
        self.add(get_ind, 1)

        return True

    def prepare_complex_trade(self, give_ind: int, get_ind: int, allow_player_trade: bool = False):
        assert self.hand[give_ind] >= 1, "Player does not have that resources"

        if allow_player_trade:
            raise NotImplementedError("P2P Trading Not Implemented")
        else:
            self.trade_pile[give_ind] += 1
            if self.trade_pile.sum() > self.best_trade_rate[get_ind]:
                raise NotImplementedError("Variable trade rates not implemented")
            elif self.trade_pile.sum() == self.best_trade_rate[get_ind]:
                self.give(self.trade_pile)
                self.add(get_ind, 1)
                self.trade_pile[:] = 0
            else:
                pass



    def rob(self):
        if sum(self.hand) > 7:
            for i in range(len(self.hand)):
                self.hand[i] -= self.hand[i] // 2

    # def sample_action(self, game, road_mask, village_mask, i_am_player):
    #     return self.agent.sample_action(road_mask, village_mask)
