# Learning algorithm
The learning algorithm used was DQN (deep Q-learning with a neural network as function approximation).
The agent act in the environment to create an understanding of what actions create the most rewards in what states.
The agent perceives the state of the environment and acts according to a policy. For each action it takes,
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
This procedure is repeated for UPDATE_EVERY (hyper parameter) steps, so the network has approximated the values fore some time.
After that the parameters of this network (local netowrk) are copied to a second network with the same structure (traget network).
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

The actions the agent choses to interact with the environment are chosen according to an epsilon-greedy way. That is
a threshold `epsilon [0,1]` is defined, that specifies the probability a random action is used. With the probability
`1-epsilon` the greedy action according to the current best approximation of the state-action values is chosen.
The value for epsilon is reduced over time by the factor 0.995 to encourage random exploration of actions at the start
and use the best actions later during the training.

The rewards received by an agent of this type can be shown in this figure:
![Best performance over all](DQN/best_model_overall/score_plot_1992.jpg)

This agent was trained 2000 episodes and reached an average reward of 15.39.

The reward > 13 was achieved after 658 episodes. The next figure shows the learning process of that agent.
![Earlies solution to the environment](DQN/earliest_success_model/score_plot_658.jpg)

# Future ideas to improve performance
To improve performance it would be nice to try out DDQN and DDQN with prioritized experience replay to train
the agent even faster on the environment. According to [this paper](https://arxiv.org/pdf/1511.05952.pdf)
prioritized experience replay would have the advantage that the experience is visited according to how much the agent
can learn from the example (TD-error). Therefore experiences that hold more knowledge for the agent will be visited
more often and therefore increase the learning efficiency of the agent.

DDQNs would, according to [this paper](https://arxiv.org/pdf/1509.06461.pdf) help reducing the inherent over estimation
of DQN networks, which comes from always picking the maximum q-value among all possible actions, even though the
q-values are still evolving. To mitigate this, DDQN uses a second function approximator to take the q-values from while
choosing an action according to the maximum value of the first function approximator. This way the estimation
between the two approximators must align in order to create a high update. Therefore luckily obtained high q-values
don't necessarily result in a high update, especially in the early stage of the training and therefore make the
training more robust.

Also probably value based methods can be used for this scenario as well. Even though it's not necessarily an
improvement in learning speed or efficiency, it would be nice to try it out and see the differences.

## Note on future ideas
During the past days, after writing the initial solution with DQN, I for fun also implemented DDQN and DDQN
with prioritized experience replay.
DDQN is not much different from DQN but I noticed that one can use a bigger update step of the target network (TAU).
That's probably because it's less over optimistic and therefore the two values of the local and target network don't
differ too much. As a result the training converges even a little bit faster.

Prioritized experience replay works pretty nicely as well, but is very slow in my vanilla implementation.
A tree structure or s.th. different with a faster search and sorting mechanism would speed up the training a lot.
So for future implementations this should be considered. Also I didn't need to change the parameters much to make
it converge. Probably with sophisticated parameter search this solution can work a lot faster.

# Best performace parameters for DQN:
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