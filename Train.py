from Environment.Game import Game
from Learner.RandomAgent import RandomAgent

game = Game([

    RandomAgent(),
    RandomAgent()

])

game.start(render=True)
