import threading
import traceback
from collections import deque
from typing import Dict

from torch import Tensor, autograd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Environment import Game
from Learner.AgentTracker import AgentTracker
from Learner.Agents.PPOAgent import PPOAgent
from Learner.Nets import PPONet
from Learner.PrioReplayBuffer import InMemBuffer
from Learner.Trainer import PPOTrainer, SACTrainer
from Learner.constants import *

autograd.set_detect_anomaly(True)
writer = SummaryWriter()

game = Game(
    n_players=N_PLAYERS,
    max_turns=MAX_TURNS
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
    'load_state': False
}

learner_net = PPONet(learner_net_init, batch_size=BATCH_SIZE)
actor_net_list = [PPONet(actor_net_init, batch_size=BATCH_SIZE) for _ in range(N_PLAYERS)]

buffer = InMemBuffer(
    alpha=REPLAY_ALPHA,
    beta=REPLAY_BETA,
    capacity=REPLAY_MEMORY_SIZE,
    max_seq_len=MAX_SEQUENCE_LENGTH,
)
trainer = PPOTrainer(
    net=learner_net,
    buffer=buffer,
    batch_size=BATCH_SIZE,
    gamma=GAMMA,
    learning_rate=LEARNING_RATE,
    reward_scale=REWARD_SCALE
)
agent_list = [
    PPOAgent(
        net=net,
        game=game,
        buffer=buffer,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        burn_in_length=BURN_IN_LENGTH
    )
    for net in actor_net_list
]
agent_tracker = AgentTracker(
    eps_min=EPS_MIN,
    eps_max=EPS_MAX,
    eps_zero=EPS_ZERO,
    eps_one=EPS_ONE
)
agent_tracker.register_agents(agent_list)
agent_tracker.load_contestants()
game.register_agents(agent_list)
game.reset()

def write_stats(stats: Dict):
    for key, value in stats.items():
        if isinstance(value, Tensor):
            writer.add_histogram(key, value, global_step=trainer.tick_iter)
        elif isinstance(value, float | int):
            writer.add_scalar(key, value, global_step=trainer.tick_iter)
        else:
            raise TypeError(f'Unexpected type {type(value)} to print to tensorboard')


def learner_loop():
    while not stop_event.is_set():
        td_loss, stats = trainer.tick()
        td_loss_hist.append(td_loss)
        # td_loss_mean = sum(td_loss_hist) / max(1, len(td_loss_hist))
        writer.add_scalar('TD Loss smooth', sum(td_loss_hist) / max(1, len(td_loss_hist)), trainer.tick_iter)
        writer.add_scalar('TD Loss', td_loss, trainer.tick_iter)
        write_stats(stats)


def actor_loop():
    iterator = tqdm(range(MAX_STEPS))
    for i in iterator:
        if stop_event.is_set():
            return
        observation = game.extract_attributes()
        current_player = game.current_player
        action = game.current_agent.sample_action(observation, i_am_player=current_player)
        reward, done, succeeded = game.step(action)
        game.player_agents[current_player].update_score(reward)
        reward += game.players[current_player].flush_reward()
        game.player_agents[current_player].update_reward(reward, done, game, current_player)

        # On episode termination
        if done:
            if game.episode % 10 == 0:
                game.render(training_img=True)
            agent_tracker.update_elo()
            titan = agent_tracker.get_titan()
            writer.add_scalar('TitanLastScore', titan.episode_score, game.episode)
            writer.add_scalar('TitanAvgScore', titan.avg_score, game.episode)
            writer.add_scalar('TitanBeatTime', titan.avg_beat_time, game.episode)
            writer.add_scalar('TitanWinMean', titan.mean_win, game.episode)
            writer.flush()
            for ii in range(N_PLAYERS):
                game.player_agents[ii].signal_episode_done(ii)
                game.player_agents[ii].clear_cache()
            agent_tracker.load_contestants('weighted')
            agent_tracker.shuffle_agents()
            game.reset()

        iterator.set_postfix_str(
            f"Ep: {game.episode}-{int(game.turn)}, "
            f"TD Loss: {sum(td_loss_hist) / max(1, len(td_loss_hist)):.3e} (tick {trainer.tick_iter:d}), "
            f"Score: {[int(player.points) for player in game.players]}, "
            f"ScoreHist: {[agent.avg_score for agent in game.player_agents]}, "
            f"WinMean: {[f'{100*agent.mean_win:.0f}' for agent in game.player_agents]}, "
            f"AvgBeatTime: {[int(agent.avg_beat_time) for agent in game.player_agents]}, "
            f"Players: {[agent for agent in game.player_agents]}"
        )
    stop_event.set()


def graceful_shutdown():
    stop_event.set()
    learner_thread.join()
    actor_thread.join()


def learner_worker():
    try:
        learner_loop()
    except Exception as e:
        stack_trace = traceback.format_exc()
        print(f"EXCEPTION STACK TRACE:{stack_trace}")
        print('EXCEPTION IN LEARNER LOOP: ', e)
    finally:
        stop_event.set()


def actor_worker():
    try:
        actor_loop()
    except Exception as e:
        stack_trace = traceback.format_exc()
        print(f"EXCEPTION STACK TRACE:{stack_trace}")
        print('EXCEPTION IN ACTOR LOOP: ', e)
    finally:
        stop_event.set()


td_loss_hist = deque(maxlen=HISTORY_DISPLAY)
td_loss_mean = 0

stop_event = threading.Event()

try:
    learner_thread = threading.Thread(target=learner_worker)
    actor_thread = threading.Thread(target=actor_worker)

    learner_thread.start()
    actor_thread.start()

    learner_thread.join()
    actor_thread.join()
except KeyboardInterrupt:
    print('Interrupted')
finally:
    graceful_shutdown()
