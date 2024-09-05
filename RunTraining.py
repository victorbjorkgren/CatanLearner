import multiprocessing
import queue
import traceback
from collections import deque
from datetime import datetime
from typing import Dict

from torch import Tensor, autograd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Environment import Game
from Learner.AgentTracker import AgentTracker
from Learner.Agents.PPOAgent import PPOAgent
from Learner.Nets import PPONet
from Learner.PrioReplayBuffer import InMemBuffer
from Learner.Trainer import PPOTrainer
from Learner.constants import *

autograd.set_detect_anomaly(True)

import subprocess

def get_git_commit_hash():
    try:
        # Running the git command to get the current commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        return commit_hash
    except subprocess.CalledProcessError:
        return "no-git"

# td_loss_hist = deque(maxlen=HISTORY_DISPLAY)

def learner_loop(stop_event, tensorboard_queue, print_queue, buffer, file_lock):
    writer = SummaryWriter(log_dir='./runs/' + datetime.now().strftime('%b%d-%y--%H-%M-%S-') + EXPERIMENT_NAME)

    def write_stats(stats: Dict, step: int):
        for key, value in stats.items():
            if isinstance(value, Tensor):
                writer.add_histogram(key, value, global_step=step)
            elif isinstance(value, float | int):
                writer.add_scalar(key, value, global_step=step)
            else:
                raise TypeError(f'Unexpected type {type(value)} to print to tensorboard')

    learner_net_init = {
        'game': Game(n_players=N_PLAYERS, max_turns=MAX_TURNS),
        'n_power_layers': N_POWER_LAYERS,
        'n_embed': N_HIDDEN_NODES,
        'n_output': N_PLAYERS,
        'on_device': LEARNER_DEVICE,
        'load_state': LOAD_Q_NET,
        'lock': file_lock,
    }
    learner_net = PPONet(learner_net_init, batch_size=BATCH_SIZE)
    trainer = PPOTrainer(
        net=learner_net,
        buffer=buffer,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        reward_scale=REWARD_SCALE
    )

    while not stop_event.is_set():
        # Learner process tick
        td_loss, stats = trainer.tick()

        # Write data to the actor process
        print_queue.put({
            'td_loss': td_loss,
            'tick_iter': trainer.tick_iter
        })

        # Write data from learner process
        stats['TD_Loss'] = td_loss
        write_stats(stats, trainer.tick_iter)

        # Write data from actor process
        while not tensorboard_queue.empty():
            try:
                data = tensorboard_queue.get_nowait()
                actor_step = data.pop('step')
                write_stats(data, actor_step)
            except queue.Empty:
                break


def actor_loop(stop_event, tensorboard_queue, print_queue, buffer, file_lock):
    game = Game(
        n_players=N_PLAYERS,
        max_turns=MAX_TURNS
    )

    actor_net_init = {
        'game': game,
        'n_power_layers': N_POWER_LAYERS,
        'n_embed': N_HIDDEN_NODES,
        'n_output': N_PLAYERS,
        'on_device': ACTOR_DEVICE,
        'load_state': False,
        'lock': file_lock,
    }

    actor_net_list = [PPONet(actor_net_init, batch_size=BATCH_SIZE) for _ in range(N_PLAYERS)]

    iterator = tqdm(total=MAX_STEPS)
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
    learner_postfix = {}

    while not stop_event.is_set():
        observation = game.extract_attributes()
        current_player = game.current_player
        action, pi = game.current_agent.sample_action(observation, i_am_player=current_player)
        if (game.episode % 1000 == 0) or (game.episode == BATCH_SIZE + 1):
            game.render_side_by_side(pi, 'testing')
        reward, done, succeeded = game.step(action)
        game.player_agents[current_player].update_score(reward)
        reward += game.players[current_player].flush_reward()
        game.player_agents[current_player].update_reward(reward, done, game, current_player)

        # On episode termination
        if done:
            iterator.update(1)
            if game.episode >= MAX_STEPS:
                stop_event.set()
            if game.episode % 10 == 0:
                game.render(render_type='training')
            agent_tracker.update_elo()
            titan = agent_tracker.get_titan()
            tensorboard_queue.put({'TitanLastScore': titan.episode_score, 'step': game.episode})
            tensorboard_queue.put({'TitanAvgScore': titan.avg_score, 'step': game.episode})
            tensorboard_queue.put({'TitanBeatTime': titan.avg_beat_time, 'step': game.episode})
            tensorboard_queue.put({'TitanWinMean': titan.mean_win, 'step': game.episode})
            for ii in range(N_PLAYERS):
                game.player_agents[ii].signal_episode_done(ii)
                game.player_agents[ii].clear_cache()
            agent_tracker.load_contestants('weighted')
            agent_tracker.shuffle_agents()
            game.reset()
        actor_postfix = {
            'Ep': f"{game.episode}-{int(game.turn)}",
            # f"TD Loss: {sum(td_loss_hist) / max(1, len(td_loss_hist)):.3e} (tick {trainer.tick_iter:d}), "
            'Score': f"{[int(player.points) for player in game.players]}",
            'ScoreHist': f"{[agent.avg_score for agent in game.player_agents]}",
            'WinMean': f"{[f'{100 * agent.mean_win:.0f}' for agent in game.player_agents]}",
            'AvgBeatTime': f"{[int(agent.avg_beat_time) for agent in game.player_agents]}"
            # f"Players: {[agent for agent in game.player_agents]}"}
        }

        # Get data from learner process
        while not print_queue.empty():
            learner_postfix.update(print_queue.get())

        iterator.set_postfix(actor_postfix | learner_postfix)

    stop_event.set()


def graceful_shutdown():
    stop_event.set()
    if learner_thread.is_alive():
        learner_thread.join()
    if actor_thread.is_alive():
        actor_thread.join()




    tensorboard_queue.close()
    print_queue.close()


def learner_worker(stop_event, tensorboard_queue, print_queue, buffer, file_lock):
    try:
        learner_loop(stop_event, tensorboard_queue, print_queue, buffer, file_lock)
    except Exception as e:
        stack_trace = traceback.format_exc()
        print(f"EXCEPTION STACK TRACE:{stack_trace}")
        print('EXCEPTION IN LEARNER LOOP: ', e)
    finally:
        stop_event.set()


def actor_worker(stop_event, tensorboard_queue, print_queue, buffer, file_lock):
    try:
        actor_loop(stop_event, tensorboard_queue, print_queue, buffer, file_lock)
    except Exception as e:
        stack_trace = traceback.format_exc()
        print(f"EXCEPTION STACK TRACE:{stack_trace}")
        print('EXCEPTION IN ACTOR LOOP: ', e)
    finally:
        stop_event.set()


if __name__ == '__main__':
    stop_event = multiprocessing.Event()
    tensorboard_queue = multiprocessing.Queue()
    print_queue = multiprocessing.Queue()
    file_lock = multiprocessing.Lock()

    buffer = InMemBuffer(
        alpha=REPLAY_ALPHA,
        beta=REPLAY_BETA,
        capacity=REPLAY_MEMORY_SIZE,
        max_seq_len=MAX_SEQUENCE_LENGTH,
    )

    args = (stop_event, tensorboard_queue, print_queue, buffer, file_lock)

    try:
        learner_thread = multiprocessing.Process(target=learner_worker, args=args)
        actor_thread = multiprocessing.Process(target=actor_worker, args=args)

        learner_thread.start()
        actor_thread.start()

        learner_thread.join()
        actor_thread.join()

        tensorboard_queue.close()
        print_queue.close()
    except KeyboardInterrupt:
        print('Interrupted')
    finally:
        graceful_shutdown()
