from collections import deque

import torch
from tqdm import tqdm

from Environment.Game import Game
from Learner.Agents import RandomAgent, QAgent
from Learner.Nets import GameNet
from Learner.PrioReplayBuffer import PrioReplayBuffer
from Learner.Train import Trainer

MAX_STEPS = 1_000_000
HISTORY_DISPLAY = 100

game = Game(
    n_players=2
)
q_net = GameNet(
    game=game,
    n_power_layers=4,
    n_embed=16,
    n_output=4
)
target_net = GameNet(
    game=game,
    n_power_layers=4,
    n_embed=16,
    n_output=4
)
trainer = Trainer(
    q_net=q_net,
    target_net=target_net,
    batch_size=128,
    gamma=.9,
    memory_size=2 ** 14,  # 16384
    alpha=.7,
    beta=.9
)


game.set_agents([
    RandomAgent(),
    QAgent(q_net, game)
])
game.reset()

reward_history = torch.zeros((HISTORY_DISPLAY, 2))
beat_time = torch.zeros((HISTORY_DISPLAY, 2))
iterator = tqdm(range(MAX_STEPS))
for i in iterator:
    observation = q_net.get_dense(game)
    action, raw_action = game.current_agent.sample_action(game, observation, i_am_player=game.current_player)
    reward, done, action = game.step(action, render=False)
    new_observation = q_net.get_dense(game)

    trainer.add(observation, raw_action, new_observation, reward, done, game.current_player)
    loss = trainer.train(i, game.episode)

    reward_history[game.episode % reward_history.shape[0], :] = reward.clamp_min(0)
    beat_time[game.episode % beat_time.shape[0], reward == 1] = game.turn

    if done:
        game.reset()

    mask = beat_time != 0
    avg_beat_time = (beat_time * mask).sum(dim=0) / mask.sum(dim=0)
    iterator.set_postfix_str(
        f"Ep: {game.episode}, "
        f"Loss: {loss:.4f}"
        f"WinHistory: {reward_history.sum(0).long().tolist()}, "
        f"AvgBeatTime: {avg_beat_time.tolist()}"
    )
