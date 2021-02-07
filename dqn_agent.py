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


class DQNAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        :param state_size: (int) dimension of each state
        :param action_size: (int) dimension of each action
        :param seed: (int) random seed
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
        """
        Adds the current state-action value to the memory and lets the agent learn if UPDATE_EVERY many steps are taken
        and the memory has more entries then BATCH_SIZE.

        :param state:       current state
        :param action:      taken action
        :param reward:      received reward
        :param next_state:  next state seen after action
        :param done:        boolean if the episode ended after the action
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % PARAM.UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > PARAM.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, PARAM.GAMMA)

    def act(self, state, eps=0.):
        """
        Returns actions for given state as per current policy.

        :param state: (array_like) current state
        :param eps: (float) epsilon, for epsilon-greedy action selection
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
        """
        Gets the state-action value of the target network. That is, the current estimate of the target network for the
        next state including the seen reward.

        :param next_states: next state for each entry in the sampled mini batch
        :param rewards:     rewards seen for each sample in the mini batch
        :param gamma:       decay factor for current estimate
        :param dones:       indicator if the episode ended for each sample in the mini batch
        :return:
        """
        # Get predicted Q values
        qtarget_values = self.qnetwork_target(next_states).detach()

        # get max of it
        best_qtarget_value = qtarget_values.max(1)

        # reduce one dimension
        best_qtarget_value = best_qtarget_value[0]

        # reshape to 2d matrix with one value in it for 1st dimension (so difference can be calculated)
        # >>> torch.unsqueeze(x, 1)
        # tensor([[ 1],
        #        [ 2],
        #        [ 3],
        #        [ 4]])
        best_qtarget_value = best_qtarget_value.unsqueeze(1)

        # use vector formulation of:
        # if dones == 1:
        #    Q_targets = rewards
        # else:
        #    Q_targets = rewards + (gamma * best_qtarget_value)
        q_targets = rewards + (gamma * best_qtarget_value * (1 - dones))

        return q_targets

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.

        :param experiences:  (Tuple[torch.Variable]) tuple of (s, a, r, s', done) tuples
        :param gamma: (float) discount factor
        """

        states, actions, rewards, next_states, dones = experiences

        Q_targets = self.get_dqg_target(next_states, rewards, gamma, dones)

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
        self.soft_update(self.qnetwork_local, self.qnetwork_target, PARAM.TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        :param local_model: (PyTorch model) weights will be copied from
        :param target_model: (PyTorch model) weights will be copied to
        :param tau: (float) interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

