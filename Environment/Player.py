###
# Player states:
#
# TODO: add cards etc.
# [Bricks, Grains, Ores, Lumbers, Wools]
#
class Player:
    def __init__(self, agent):
        self.hand = [0, 0, 0, 0, 0]
        self.points = 0
        self.agent = agent

    @property
    def state(self):
        return self.hand + [self.points]

    def add(self, ind, n):
        self.hand[ind] += n

    def sub(self, ind, n):
        self.hand[ind] -= n

    def rob(self):
        if sum(self.hand) > 7:
            for i in range(len(self.hand)):
                self.hand[i] -= self.hand[i] // 2

    def sample_action(self):
        return self.agent.sample_action()

    def __str__(self):
        # [Bricks, Grains, Ores, Lumbers, Wools]
        return ((f"Brick: {self.hand[0]:0d}, "
                 f"Grain: {self.hand[1]}, "
                 f"Ore: {self.hand[2]}, "
                 f"Lumber: {self.hand[3]}, "
                 f"Wool: {self.hand[4]}")
                + f"\nPoints: {self.points}")
