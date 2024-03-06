from collections import deque
import threading

from tqdm import tqdm

from Environment import Game
from Learner.AgentTracker import AgentTracker
from Learner.Agents import QAgent
from Learner.Nets import GameNet
from Learner.PrioReplayBuffer import PrioReplayBuffer
from Learner.Trainer import Trainer
from Learner.constants import *


game = Game(
    n_players=N_PLAYERS,
    max_turns=300
)
learner_net_init = {
    'game': game,
    'n_power_layers': N_POWER_LAYERS,
    'n_embed': N_HIDDEN_NODES,
    'n_output': N_PLAYERS,
    'on_device': LEARNER_DEVICE,
    'load_state': LOAD_Q_NET
}
actor_net_init = {
    'game': game,
    'n_power_layers': N_POWER_LAYERS,
    'n_embed': N_HIDDEN_NODES,
    'n_output': N_PLAYERS,
    'on_device': ACTOR_DEVICE,
    'load_state': LOAD_Q_NET
}

learner_q_net = GameNet(learner_net_init).to(LEARNER_DEVICE)
target_q_net = GameNet(learner_net_init).to(LEARNER_DEVICE)
target_q_net.clone_state(learner_q_net)
actor_q_net_list = [GameNet(actor_net_init).to(ACTOR_DEVICE) for _ in range(N_PLAYERS)]

buffer = PrioReplayBuffer(
    alpha=REPLAY_ALPHA,
    beta=REPLAY_BETA,
    capacity=REPLAY_MEMORY_SIZE,
    max_seq_len=MAX_SEQUENCE_LENGTH
)
trainer = Trainer(
    q_net=learner_q_net,
    target_net=target_q_net,
    buffer=buffer,
    batch_size=BATCH_SIZE,
    gamma=GAMMA,
)
agent_list = [
    QAgent(
        q_net=net,
        game=game,
        buffer=buffer,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        burn_in_length=BURN_IN_LENGTH
    )
    for net in actor_q_net_list
]
agent_tracker = AgentTracker(
    eps_min=EPS_MIN,
    eps_max=EPS_MAX,
    eps_zero=EPS_ZERO,
    eps_one=EPS_ONE
)
agent_tracker.register_agents(agent_list)
game.register_agents(agent_list)
game.reset()


def learner_loop():
    while True:
        td_loss = trainer.tick()
        td_loss_hist.append(td_loss)


td_loss_hist = deque(maxlen=HISTORY_DISPLAY)

learner_thread = threading.Thread(target=learner_loop)
learner_thread.start()

iterator = tqdm(range(MAX_STEPS))
for i in iterator:
    observation, obs_player = actor_q_net_list[0].get_dense(game)
    action, raw_action = game.current_agent.sample_action(observation, i_am_player=game.current_player)
    reward, done, succeeded = game.step(action)
    game.player_agents[obs_player].update_reward(reward, done, obs_player)

    # On episode termination
    if done:
        game.render(training_img=True)
        agent_tracker.update_elo()
        for ii in range(N_PLAYERS):
            game.player_agents[ii].signal_episode_done(ii)
            game.player_agents[ii].clear_cache()
        agent_tracker.load_contestants('weighted')
        agent_tracker.shuffle_agents()
        game.reset()

    iterator.set_postfix_str(
        f"Ep: {game.episode}-{int(game.turn)}, "
        f"TD Loss: {sum(td_loss_hist) / HISTORY_DISPLAY:.3e} (tick {trainer.tick_iter:d}), "
        f"Score: {[int(player.points) for player in game.players]}, "
        f"WinHist: {[agent.sum_win for agent in game.player_agents]}, "
        f"AvgBeatTime: {[int(agent.avg_beat_time) for agent in game.player_agents]}, "
        f"Players: {[agent for agent in game.player_agents]}"
    )
