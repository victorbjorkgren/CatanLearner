import os

import torch as T
from tqdm import tqdm
import atexit

from Environment import Game
from Learner.Agents import RandomAgent, QAgent
from Learner.Nets import GameNet
from Learner.Train import Trainer

MAX_STEPS = 100_000_000
HISTORY_DISPLAY = 100
N_PLAYERS = 2

DRY_RUN = 0
BATCH_SIZE = 64

REPLAY_MEMORY_SIZE = 2 ** 13  # 8192
REPLAY_ALPHA = .7
REPLAY_BETA = .8

REWARD_MIN_FOR_Q = 3

LOAD_Q_NET = True
LOAD_BUFFER = False

device = 'cuda' if T.cuda.is_available() else 'cpu'
# device = 'cpu'

game = Game(
    n_players=N_PLAYERS,
    max_turns=500
)
q_net = GameNet(
    game=game,
    n_power_layers=4,
    n_embed=16,
    n_output=N_PLAYERS,
    on_device=device,
    load_state=LOAD_Q_NET
)
target_net = GameNet(
    game=game,
    n_power_layers=4,
    n_embed=16,
    n_output=N_PLAYERS,
    on_device=device,
    load_state=LOAD_Q_NET
)
trainer = Trainer(
    q_net=q_net,
    target_net=target_net,
    batch_size=BATCH_SIZE,
    dry_run=DRY_RUN,
    reward_min=REWARD_MIN_FOR_Q,
    gamma=.9,
)
target_net.clone_state(q_net)
q_net = q_net.to(device)
target_net = target_net.to(device)

random_agent = RandomAgent(
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

# random_vs_random = [random_agent, random_agent]
random_vs_q = [random_agent, q_agent]
# q_vs_q = [q_agent, q_agent]

trainer.register_agents([random_agent, q_agent])
game.register_agents(random_vs_q)
game.reset()

td_loss, rule_loss = 0, 0
reward_history = T.zeros((HISTORY_DISPLAY, 2))
beat_time = T.zeros((HISTORY_DISPLAY, 2))
td_loss_hist = T.zeros((HISTORY_DISPLAY,))
rule_loss_hist = T.zeros((HISTORY_DISPLAY,))
iterator = tqdm(range(MAX_STEPS))
for i in iterator:
    observation, obs_player = q_net.get_dense(game)
    action, raw_action = game.current_agent.sample_action(
        game,
        observation,
        i_am_player=game.current_player,
        remember=True
    )
    reward, done, succeeded = game.step(action)

    # Should not trigger
    if not succeeded:
        print(f'Invalid action {raw_action.tolist()} by player {obs_player}')
        raw_action = T.tensor((73, 73))

    # Training tick
    if reward_history.sum() > 0:
        td_loss, rule_loss = trainer.train()

    # Update trackers
    reward_history[game.episode % reward_history.shape[0], :] = reward.clamp_min(0)
    beat_time[game.episode % beat_time.shape[0], reward == 1] = game.turn
    td_loss_hist[i % td_loss_hist.shape[0]] = td_loss
    rule_loss_hist[i % rule_loss_hist.shape[0]] = rule_loss

    # On episode termination
    if done:
        for ii in range(len(game.player_agents)):
            game.player_agents[ii].update_reward(reward[ii])
        game.render(training_img=True)
        q_net.save()
        game.reset()

    mask = T.tensor(beat_time != 0)
    avg_beat_time = (beat_time * mask).sum(dim=0) / mask.sum(dim=0)
    iterator.set_postfix_str(
        f"Ep: {game.episode}-{int(game.turn)}, "
        f"TD Loss: {td_loss_hist[td_loss_hist.nonzero()].mean().item():.3e} "
        f"Rule Loss: {rule_loss_hist[rule_loss_hist.nonzero()].mean().item():.3e} "
        f"WinHistory: {reward_history.sum(0).long().tolist()}, "
        f"AvgBeatTime: {avg_beat_time.int().tolist()}, "
    )
