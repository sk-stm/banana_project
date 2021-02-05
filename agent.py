import numpy as np
import random
from model import QNetwork
from prioritized_memory import PriorityMemory
from replay_buffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.memory = PriorityMemory(20000)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        if type(self.memory) == ReplayBuffer:
            self.memory.add(state, action, reward, next_state, done)
        elif type(self.memory) == PriorityMemory:
            with torch.no_grad():
                np_state = torch.from_numpy(state).float().unsqueeze(0).to(device)
                current_value = self.qnetwork_local(np_state)[0][action]

                target_value = self.qnetwork_target(np_state)
                updated_value = reward + GAMMA * torch.max(target_value)

                error = abs(current_value - updated_value)
                self.memory.add(error, (state, action, reward, next_state, done))
        else:
            raise NotImplemented()

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:

            # If enough samples are available in memory, get random subset and learn
            if type(self.memory) == ReplayBuffer:
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

            elif type(self.memory) == PriorityMemory:
                if self.memory.get_length() > BATCH_SIZE:
                    experiences = self.memory.sample(BATCH_SIZE)
                    self.learn(experiences, GAMMA)

            else:
                raise NotImplemented()

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def get_dqg_target(self, next_states, rewards, gamma, dones):

        # Get predicted Q values
        q_hat = self.qnetwork_target(next_states).detach()
        # print(q_hat)

        # get max of it
        q_hat = q_hat.max(1)
        # print(q_hat)

        # reduce one dimension
        q_hat = q_hat[0]
        # print(q_hat)

        # reshape to 2d matrix with one value in it for 1st dimension (so difference can be calculated)
        # >>> x = torch.tensor([1, 2, 3, 4])
        # >>> torch.unsqueeze(x, 0)
        # tensor([[ 1,  2,  3,  4]])
        # >>> torch.unsqueeze(x, 1)
        # tensor([[ 1],
        #        [ 2],
        #        [ 3],
        #        [ 4]])
        q_hat = q_hat.unsqueeze(1)

        # doesn't work due to dones being a vector.
        # if dones == 1:
        #    Q_targets = rewards
        # else:
        #    Q_targets = rewards + (gamma * q_hat)

        # taken from solution. nice matrix formulation of if statement
        Q_targets = rewards + (gamma * q_hat * (1 - dones))

        return Q_targets

    def get_ddqn_targets(self, next_states, rewards, gamma, dones):
        q_hat = self.qnetwork_target(next_states).detach()
        # print(f"q_hat: {q_hat}")
        ddq_hat = self.qnetwork_local(next_states).detach()
        # print(f"ddq_hat: {ddq_hat}")
        ddq_hat = q_hat.argmax(1)
        # print(f"argmax ddq_hat: {ddq_hat}")
        q_hat = q_hat.index_select(1, ddq_hat)[:, 0]
        # print(f"max of dqn according to ddqn: {q_hat}")
        q_hat = q_hat.unsqueeze(1)
        Q_targets = rewards + (gamma * q_hat * (1 - dones))

        return Q_targets

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        if type(self.memory) == ReplayBuffer:
            states, actions, rewards, next_states, dones = experiences
        elif type(self.memory) == PriorityMemory:

        else:
            raise NotImplemented()

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"
        # print(f"actions: {actions}")

        # taken from solution because it was impossible to debug this. Kernel of notebook needs to be restarts each and every time!

        ########## DQN
        # Q_targets = self.get_dqg_target(next_states, rewards, gamma, dones)

        ######### DDQN
        Q_targets = self.get_ddqn_targets(next_states, rewards, gamma, dones)

        # Get expected Q values
        q_exp = self.qnetwork_local(states)
        # print(q_exp)

        # gets the q values along dimention 1 according to the actions, which is used as index
        # >>> t = torch.tensor([[1,2],[3,4]])
        # >>> torch.gather(t, 1, torch.tensor([[0],[1]]))
        # tensor([[ 1],
        #        [ 4]])
        q_exp = q_exp.gather(1, actions)
        # print(q_exp)

        # compute loss
        loss = F.mse_loss(q_exp, Q_targets)

        # reset optimizer gradient
        self.optimizer.zero_grad()
        # do backpropagation
        loss.backward()
        # do optimize step
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

