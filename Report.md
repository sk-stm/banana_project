# Learning algorithm
The learning algorithm used was DQN (deep Q-learning with a neural network as function approximation).
The agent act in the environment to create an understanding of what actions create the most rewards in what states.
Therefore the agent percieves the state of the environment and acts according to a policy. For each action it takes,
it receives an reward and observes the next states it transitions to. Also it observes if the episode ended or not.

Each observation (state, action, reward, next state, done) is stored in a list, the replay memory. Once the memory
contains enough entries, a random sample of mini batch size is taken from the memory. These samples are used to train
a function approximator, in our case a fully connected neural network, specifically an MLP.

**The network has the following structure:**
```
Fully connected layer 1:  input-nodes: state_size==37, output-nodes: 128)
Fully connected layer 1:  input-nodes: 128, output-nodes: 64)
Fully connected layer 1:  input-nodes: 64, output-nodes: action_size==4)
```

This network (local network) learns to approximate the value for each action of one state.
This procedure is repeated for UPDATE_EVERY steps, so the network has approximated the values fore some time.
After that the parameters of the local netowrk are copied to a second network with the same structure (traget network).
The target network is a network that parameters are kept constant during the training. By copying the parameters
from the local network to the target network, it captures the current state of the overall training. This network
is then used to estimate the total discounted reward for the next state.

That is, for each state the value of the next state is estimated to approximate the value of the current state a little
bit better. Why is that necessary?

The local network is trained in a supervised manner. That is, for each state there are target values that the network
aims towards. They can be used to create a loss for the network to optimize its parameters.

Specifically this target is the estimated total discounted reward for the next state from the target network:
if the episode ended, the target == reward, else the target is:
targets = rewards + (gamma * best_qtarget_value)

Where best_qtarget_value is the best value for the next state that the target network contains. And `gamma == 0.99` is the
discount factor for using the estimated target value. This is useful because the target is only estimated and also changes
over time. The highter gamma, the more it is used to update. The lower gamma, the more the reward i trusted, and the less
the estimated value of the next state is used.

The network is trained with Adam optimizer, which is a standard first good choice of optimizers because it inherently
accounts for the update momentum and other update properties. The **learning rate** was set to `LR=0.0005`.
The update step of the parameters of the target network is done softly. That is, the parameters are not just copied
over from the local network, but are softly updated according to `TAU==0.003`. That means that the old parameters of
the target network are preserved by a factor of `1-TAU == 0.997` and updated by the local network by `TAU==0.003`.
This way the target network doesn't change too quickly and the target state-action values stay different from the local
state-action values, to create a meaningful loss. The **batch size** for training samples was set to `BATCH_SIZE==64`


So with each learning step the local network approximates the current step a little bit better because it is driven
towards the reward + the expected reward at the next state according to the target network. The target network is
updated with the better approximation the local network gained after some steps, by copying its parameters.

This is repeated until a good approximation of state-action values is achieved.

Tha actions the agent choses to interact with the environment are chosen according to an epsilon-greedy way. That is
a threshold `epsilon [0,1]` is defined, that specifies the probability a random action is used. With the probability
`1-epsilon` the greedy action according to the current best approximation of the state-action values is chosen.
The value for epsilon is reduced over time by the factor 0.995 to encourage random exploration of actions at the start
and use the best actions later during the training.

The rewards received by an agent of this type can be shown in this figure:
# TODO include picture
[DQN/best_model_overall/score_plot_1992.jpg]
This agent was trained 2000 episodes and reached an average reward of 15.39.

The reward > 13 was achieved after 658 episodes. The next figure shows the learning process of that agent.
# TODO include figure
[DQN/earliest_success_model/score_plot_658.jpg]

# Future ideas to improve performance
# TODO formulate
- prioritized experience replay
- DDQN
- try A2C or A3C


# TODO remove this or better put it somewhere else.
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