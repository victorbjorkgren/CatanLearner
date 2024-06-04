import torch as T

# ENVIRONMENT AND DISPLAY
MAX_STEPS = 100_000_000
HISTORY_DISPLAY = 1_000
MAX_TURNS = 800
N_PLAYERS = 2

# LEARNER
BATCH_SIZE = 8
LEARNING_RATE = 3e-5
REWARD_SCALE = .1
GAMMA = .999
LOSS_CLIP = .6
GRAD_CLIP = .5

# ACTORS
USE_ACTOR_PRIO = True
FAILURE_ALLOWANCE = 2
EPS_MIN = .9
EPS_MAX = .99
EPS_ZERO = .5  # Chance that epsilon becomes zero
EPS_ONE = .1  # Chance that epsilon becomes one

# NETWORK
N_POWER_LAYERS = 2
N_HIDDEN_NODES = 32
MAX_SEQUENCE_LENGTH = 100
BURN_IN_LENGTH = 20  # TODO: NOT USED, USE.

# PPO
PPO_VALUE_COEF = .5
PPO_ENTROPY_COEF = .0025

# REPLAY
REPLAY_MEMORY_SIZE = 2 ** 13  # 8192
REPLAY_ALPHA = .7
REPLAY_BETA = .8

LOAD_Q_NET = True
LOAD_BUFFER = True

LEARNER_DEVICE = 'cuda' if T.cuda.is_available() else 'cpu'
ACTOR_DEVICE = 'cpu'
