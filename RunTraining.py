import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from collections import deque
import torch as T
from tqdm import tqdm

from Environment import Game
from Learner.Agents import RandomAgent, QAgent
from Learner.Nets import GameNet
from Learner.Train import Trainer

# ENVIRONMENT AND DISPLAY
MAX_STEPS = 100_000_000
HISTORY_DISPLAY = 1_000
N_PLAYERS = 2

# LEARNER
REWARD_MIN_FOR_LEARNING = 0
DRY_RUN = 1024
BATCH_SIZE = 8
GAMMA = .99

# NETWORK
N_POWER_LAYERS = 2
N_HIDDEN_NODES = 32

# REPLAY
REPLAY_MEMORY_SIZE = 2 ** 10  # 1024
REPLAY_ALPHA = .9
REPLAY_BETA = .4

LOAD_Q_NET = True
LOAD_BUFFER = False

device = 'cuda' if T.cuda.is_available() else 'cpu'
# device = 'cpu'

game = Game(
    n_players=N_PLAYERS,
    max_turns=300
)
q_net = GameNet(
    game=game,
    n_power_layers=N_POWER_LAYERS,
    n_embed=N_HIDDEN_NODES,
    n_output=N_PLAYERS,
    on_device=device,
    load_state=LOAD_Q_NET
)
target_net = GameNet(
    game=game,
    n_power_layers=N_POWER_LAYERS,
    n_embed=N_HIDDEN_NODES,
    n_output=N_PLAYERS,
    on_device=device,
    load_state=LOAD_Q_NET
)
trainer = Trainer(
    q_net=q_net,
    target_net=target_net,
    batch_size=BATCH_SIZE,
    dry_run=DRY_RUN,
    reward_min=REWARD_MIN_FOR_LEARNING,
    gamma=GAMMA,
)
target_net.clone_state(q_net)
q_net = q_net.to(device)
target_net = target_net.to(device)

random_agent = RandomAgent(
    game=game,
    capacity=REPLAY_MEMORY_SIZE,
    alpha=REPLAY_ALPHA,
    beta=REPLAY_BETA
)
q_agent = QAgent(
    q_net=q_net,
    game=game,
    capacity=REPLAY_MEMORY_SIZE,
    alpha=REPLAY_ALPHA,
    beta=REPLAY_BETA
)

agent_list = [q_agent, random_agent]
trainer.register_agents(agent_list)
game.register_agents(agent_list)
game.reset()

td_loss, rule_loss = 0, 0
td_loss_hist = deque(maxlen=HISTORY_DISPLAY)
iterator = tqdm(range(MAX_STEPS))
for i in iterator:
    observation, obs_player = q_net.get_dense(game)
    action, raw_action = game.current_agent.sample_action(observation, i_am_player=game.current_player)
    reward, done, succeeded = game.step(action)
    game.player_agents[obs_player].update_reward(reward, done, obs_player)

    # Training tick
    td_loss = trainer.train()
    td_loss_hist.append(td_loss)

    # On episode termination
    if done:
        for ii in range(N_PLAYERS):
            game.player_agents[ii].signal_episode_done(ii)
            game.player_agents[ii].clear_cache()
        game.render(training_img=True)
        agent_list.reverse()
        trainer.register_agents(agent_list)
        game.register_agents(agent_list)
        game.reset()
        q_net.save()

    iterator.set_postfix_str(
        f"Ep: {game.episode}-{int(game.turn)}, "
        f"TD Loss: {sum(td_loss_hist) / HISTORY_DISPLAY:.3e}, "
        f"RewHist: {[f'{agent.avg_reward:4f}' for agent in game.player_agents]}, "
        f"WinHist: {[agent.sum_win for agent in game.player_agents]}, "
        f"AvgBeatTime: {[agent.avg_beat_time for agent in game.player_agents]}, "
        f"Players: {[agent for agent in game.player_agents]}"
    )
