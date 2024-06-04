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

        # Add resources for first turn
        self.add(0, 4)
        self.add(1, 2)
        self.add(3, 4)
        self.add(4, 2)

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

    def sample_action(self, board, players, i_am_player):
        return self.agent.sample_action(board, players, i_am_player)

    def __str__(self):
        # [Bricks, Grains, Ores, Lumbers, Wools]
        return ((f"{int(self.hand[0])}   Brick\n"
                 f"{int(self.hand[1])}   Grain\n"
                 f"{int(self.hand[2])}     Ore\n"
                 f"{int(self.hand[3])}  Lumber\n"
                 f"{int(self.hand[4])}    Wool\n")
                + f"\n{int(self.points)}   Points")
