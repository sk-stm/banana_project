import numpy as np
import random
from model import QNetwork
from prioritized_memory import PriorityMemory
from replay_buffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim
import PARAMETERS as PARAM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDQNAgent:
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
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=PARAM.LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, PARAM.BUFFER_SIZE, PARAM.BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % PARAM.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > PARAM.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, PARAM.GAMMA)

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

    def get_ddqn_targets(self, next_states, rewards, gamma, dones):
        # get best action according to online value function approximation
        q_online = self.qnetwork_local(next_states).detach()
        q_online = q_online.argmax(1)

        # get value of target function at position of best online action
        q_target = self.qnetwork_target(next_states).detach()
        q_target = q_target.index_select(1, q_online)[:, 0]

        # reshape
        q_target = q_target.unsqueeze(1)

        # calculate more correct q-value given the current reward
        Q_targets = rewards + (gamma * q_target * (1 - dones))

        return Q_targets

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences

        Q_targets = self.get_ddqn_targets(next_states, rewards, gamma, dones)

        # Get expected Q values
        q_exp = self.qnetwork_local(states)

        # gets the q values along dimention 1 according to the actions, which is used as index
        # >>> t = torch.tensor([[1,2],[3,4]])
        # >>> torch.gather(t, 1, torch.tensor([[0],[1]]))
        # tensor([[ 1],
        #        [ 4]])
        q_exp = q_exp.gather(1, actions)

        # compute loss
        loss = F.mse_loss(q_exp, Q_targets)

        # reset optimizer gradient
        self.optimizer.zero_grad()
        # do backpropagation
        loss.backward()
        # do optimize step
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        # according to the algorithm in https://proceedings.neurips.cc/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf
        # one should update randomly in ether direction
        update_direction = np.random.binomial(1, 0.5)
        if update_direction == 0:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, PARAM.TAU)
        else:
            self.soft_update(self.qnetwork_target, self.qnetwork_local, PARAM.TAU)

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

