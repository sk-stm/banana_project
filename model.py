import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        # self.conv1 = nn.Conv2d(3, 32, kernel_size = 8, stride = 4)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # print(state)
        # x = F.relu(self.conv1(state))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
