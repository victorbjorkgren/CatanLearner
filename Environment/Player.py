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
        self.n_villages = 0
        self.n_roads = 0
        self.agent = agent

        # Add resources for first turn
        self.add(0, 4)
        self.add(1, 2)
        self.add(3, 4)
        self.add(4, 2)

    @property
    def state(self):
        return T.cat((self.hand, T.tensor([self.points])))

    def add(self, ind: int, n: int):
        self.hand[int(ind)] += int(n)

    def sub(self, ind: int, n: int):
        self.hand[ind] -= n

    def rob(self):
        if sum(self.hand) > 7:
            for i in range(len(self.hand)):
                self.hand[i] -= self.hand[i] // 2

    def sample_action(self, game, road_mask, village_mask, i_am_player):
        return self.agent.sample_action(road_mask, village_mask)

    def __str__(self):
        # [Bricks, Grains, Ores, Lumbers, Wools]
        return ((f"Brick {int(self.hand[0])}\n"
                 f"Grain {int(self.hand[1])}\n"
                 f"Ore {int(self.hand[2])}\n"
                 f"Lumber {int(self.hand[3])}\n"
                 f"Wool {int(self.hand[4])}\n")
                + f"\nPoints {int(self.points)}")
