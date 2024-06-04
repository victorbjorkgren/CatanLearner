from Environment.Game import Game
from Learner.Agents import RandomAgent, QAgent
from Learner.Nets import GameNet

# from torchrl.data import PrioritizedReplayBuffer, ListStorage

REPLAY_CAPACITY = 100000
REPLAY_ALPHA = .7
REPLAY_BETA = .9

# replay_buffer = PrioritizedReplayBuffer(alpha=REPLAY_ALPHA, beta=REPLAY_BETA, storage=ListStorage(REPLAY_CAPACITY))


game = Game(2)
agents = [
    RandomAgent(),
    QAgent(
        GameNet(game=game,
                n_power_layers=4,
                n_embed=16,
                n_output=4),
        game.board.state.edge_index)
]
game.set_agents(agents)
game.reset()
game.start(render=True)

# q_agent = QAgent(q_net)
# q_agent.sample_action(game, 0)
