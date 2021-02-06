# REINFORCEMENT LEARNING FOR UNITY BANANA PROJECT

## Project description

An agent has to navigate and act in an simulated environment consisting of a room filled with yellow and blue bananas.
The objective is to write a program that navigates the agend through the environment collecting yellow banans.

The environment is written in UNITY which can be observed through a continuous 37 dimensional statespace. The statespace contains information about the position of the agent in the environment, and about object. in its field of view. Thus the world is at agiven time only partially observed by the agent.
Additionally the world keeps changing over time as new bananas drop inside the world from time to time and the agent removes / collects bananas from the environment when it moves near them.

 The agent can perform 4 discrete actions:

- 0 - walk forward
- 1 - walk backward
- 2 - turn left
- 3 - turn right

When the agent moves near a banana this banana is collected and will give a reward of `+1` if the banana is yell and `-1` if the banana is blue.

## Project goal
The environment is considered solved if the agent get an **average reward of >=13** in 100 consecutive runs.

## Installation

1. Create virtual python environment and source it:
    - `python3.6 -m venv p1_env`
    - `source p1_env/bin/activate`
    - `pip install -U pip`
2. Clone repository:
    - `git clone https://github.com/sk-stm/banana_project.git`
    - `cd banana_project`
    - `pip install -r requirements.txt`
3. Follow the instructions to perform a minimal install of the environment for you system in https://github.com/openai/gym#id5

## Run training:
1. Run main.py and the agent start training in the environment.

## Run inference:
 TODO

## Experiments:
After making sure that all implementations work, (DQN, DDQG, DDQN with prioritized experience replay)
I tuned the hyper parameters. I observed that for all experiments, that the average score rises until a certain point
(usually 300 - 700 episodes) and then decline again.

I print epsilon along with all the values to see if there is a connection.
### Best performace yet:
agent hyper parameters
- N_EPISODES = 2000  # how many episodes to train
- MAX_T = 10000  # maximum steps per episode
- EPS_START = 1.0  # start values of epsilon (for epsilon greedy exploration)
- EPS_END = 0.01  # minimum value of epsilon
- EPS_DECAY = 0.995  # decay rate of epsilon new_eps = old_eps * eps_decay for each step
- GAMMA = 0.99  # discount factor

neural network hyper parameters
- TAU = 1e-3  # for soft update of target parameters
- LR = 5e-4  # learning rate
- UPDATE_EVERY = 4  # how often to update the network
- BATCH_SIZE = 64  # minibatch size

replay memory hyper parameters
- BUFFER_SIZE = int(1e4)  # replay buffer size

environment hyper parameters
- STATE_SIZE = 37
- ACTION_SIZE = 4

score: ~ 12.7 after around 500 episodes. epsilon ~ 0.06
After that it declined again until episode 650 to ~ 11.8 and then increased again to 13.4 at ~ 670 episosed

This is weird because the same parameters and agent declined in a previous experiment after reaching
an average score of 7.x after a couple of episodes again.

I don't have the complete overview over all the outcomes. Probably it's best to create an automatic save to
all the experiments + hyper parameters to make it reproducible and tune the parameters according to a scema.




