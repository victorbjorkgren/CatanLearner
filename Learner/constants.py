import torch as T

# ENVIRONMENT AND DISPLAY
MAX_STEPS = 100_000_000
HISTORY_DISPLAY = 1_000
N_PLAYERS = 2

# LEARNER
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
REWARD_SCALE = .1
GAMMA = .99

# ACTORS
USE_ACTOR_PRIO = False
FAILURE_ALLOWANCE = 2
EPS_MIN = .9
EPS_MAX = .99
EPS_ZERO = .3  # Chance that epsilon becomes zero
EPS_ONE = .2  # Chance that epsilon becomes one

# NETWORK
N_POWER_LAYERS = 2
N_HIDDEN_NODES = 32
MAX_SEQUENCE_LENGTH = 100
BURN_IN_LENGTH = 20

# REPLAY
REPLAY_MEMORY_SIZE = 2 ** 11  # 2048
REPLAY_ALPHA = .9
REPLAY_BETA = .4

LOAD_Q_NET = True
LOAD_BUFFER = True

LEARNER_DEVICE = 'cuda' if T.cuda.is_available() else 'cpu'
ACTOR_DEVICE = 'cpu'
