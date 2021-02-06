# agent hyper parameters
N_EPISODES = 2000  # how many episodes to train
MAX_T = 10000  # maximum steps per episode
EPS_START = 1.0  # start values of epsilon (for epsilon greedy exploration)
EPS_END = 0.01  # minimum value of epsilon
EPS_DECAY = 0.995  # decay rate of epsilon new_eps = old_eps * eps_decay for each step
GAMMA = 0.99  # discount factor

# neural network hyper parameters
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
BATCH_SIZE = 64  # minibatch size

# replay memory hyper parameters
BUFFER_SIZE = int(1e4)  # replay buffer size

# environment hyper parameters
STATE_SIZE = 37
ACTION_SIZE = 4
