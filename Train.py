from Environment.Game import Game
from Learner.Agents import RandomAgent
from Learner.Nets import GameNet

# from torchrl.data import PrioritizedReplayBuffer, ListStorage

REPLAY_CAPACITY = 100000
REPLAY_ALPHA = .7
REPLAY_BETA = .9

# replay_buffer = PrioritizedReplayBuffer(alpha=REPLAY_ALPHA, beta=REPLAY_BETA, storage=ListStorage(REPLAY_CAPACITY))


game = Game([

    RandomAgent(),
    RandomAgent()

])

q_net = GameNet(
    game=game,
    n_power_layers=4,
    n_embed=16,
    n_output=4
)

q_net(game)

# game.start(render=False)

