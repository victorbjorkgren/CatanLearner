import torch as T
from tqdm import tqdm

from Environment import Game
from Learner.Agents import RandomAgent, QAgent
from Learner.Nets import GameNet
from Learner.Train import Trainer

MAX_STEPS = 100_000_000
HISTORY_DISPLAY = 100
N_PLAYERS = 2

REWARD_MIN_FOR_Q = 3

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
    on_device=device
)
target_net = GameNet(
    game=game,
    n_power_layers=4,
    n_embed=16,
    n_output=N_PLAYERS,
    on_device=device
)
trainer = Trainer(
    q_net=q_net,
    target_net=target_net,
    batch_size=64,
    dry_run=0,
    reward_min=REWARD_MIN_FOR_Q,
    gamma=.9,
    memory_size=2 ** 14,  # 16384
    alpha=.7,
    beta=.9
)
try:
    q_net.load()
except:
    print('Weights not loaded - starting fresh')
target_net.clone_state(q_net)
q_net = q_net.to(device)
target_net = target_net.to(device)

random_vs_random = [RandomAgent(), RandomAgent()]
random_vs_q = [
    RandomAgent(),
    QAgent(
        q_net=q_net,
        game=game
    )
]
q_vs_q = [
    QAgent(
        q_net=q_net,
        game=game
    ),
    QAgent(
        q_net=q_net,
        game=game
    )
]

game.set_agents(random_vs_random)
game.reset()

reward_history = T.zeros((HISTORY_DISPLAY, 2))
beat_time = T.zeros((HISTORY_DISPLAY, 2))
loss_hist = T.zeros((HISTORY_DISPLAY,))
iterator = tqdm(range(MAX_STEPS))
for i in iterator:
    observation, obs_player = q_net.get_dense(game)
    action, raw_action = game.current_agent.sample_action(game, observation, i_am_player=game.current_player)
    reward, done, succeeded = game.step(action)
    new_observation, _ = q_net.get_dense(game)

    if not succeeded:
        raw_action = T.tensor((73, 73))

    trainer.add(observation, raw_action, new_observation, reward, done, obs_player)
    loss = trainer.train(i, game.episode)

    reward_history[game.episode % reward_history.shape[0], :] = reward.clamp_min(0)
    beat_time[game.episode % beat_time.shape[0], reward == 1] = game.turn
    loss_hist[i % loss_hist.shape[0]] = loss

    if done:
        if trainer.reward_sum >= REWARD_MIN_FOR_Q:
            game.render(training_img=True)
            game.set_agents(random_vs_q)
            q_net.save()
        else:
            game.set_agents(random_vs_random)
        game.reset()

    mask = T.tensor(beat_time != 0)
    avg_beat_time = (beat_time * mask).sum(dim=0) / mask.sum(dim=0)
    iterator.set_postfix_str(
        f"Ep: {game.episode}-{int(game.turn)}, "
        f"Loss: {loss_hist[loss_hist.nonzero()].mean().item():.3e} "
        f"WinHistory: {reward_history.sum(0).long().tolist()} ({int(trainer.reward_sum.item())}), "
        f"AvgBeatTime: {avg_beat_time.int().tolist()}, "
        f"Players: {game.player_agents[0]} vs {game.player_agents[1]}"
    )
