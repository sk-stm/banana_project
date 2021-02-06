# taken fromhttps://github.com/rlcode/per/blob/master/prioritized_memory.py

import random
import numpy as np
from priority_tree import SumTree
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PriorityMemory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01  # regularization parameter to prevent experiences to starve to death when their priority shrinks
    a = 0.6  # controles how much according to the priority is sampled a = 0 -> uniform distribution, a = 1 -> priority only
    beta = 0.4  # controls how much the weight update is corrected. b = 0 -> no correction, b = 1 -> full correction
    beta_increment_per_sampling = 0.001  # increment of beta to converge towards full correction while converging

    def __init__(self, capacity, batch_size):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.batch_size = batch_size

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def get_length(self):
        return self.tree.total()

    def sample(self):
        batch = []
        idxs = []
        segment = self.tree.total() / self.batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(self.batch_size):
            # it seems this is always a consecutive part of the memory, which wherefore has an inherent bias.
            # TODO see if that bias is a problem or not

            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        #batch = [[state, action, reward, next_state, done], [state2, action2, reward2, next_state2, done2], ...]
        batch = np.array(batch).transpose()
        #batch = [[state1, state2, ...], [action1, action2, ...], [reward1, reward2, ...],[next_state1, next_state2, ...], [done1, done2, ...]]

        # TODO my implementation for fit with numpy matrices
        states = torch.from_numpy(np.vstack(batch[0])).float().to(device)
        actions = torch.from_numpy(np.vstack(batch[1])).long().to(device)
        rewards = torch.from_numpy(np.vstack(batch[2])).float().to(device)
        next_states = torch.from_numpy(np.vstack(batch[3])).float().to(device)
        dones = torch.from_numpy(np.vstack(batch[4]).astype(np.uint8)).float().to(device)


        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return states, actions, rewards, next_states, dones, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)