from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn import functional
class TwoLayerAllConnetedNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self._in_dim = in_dim
        self.mid_layer = max(16, 2*in_dim)
        self.layers = nn.Sequential(nn.Linear(in_dim, self.mid_layer),
            # nn.BatchNorm1d(self.mid_layer),
            nn.ReLU(),
            nn.Linear(self.mid_layer, out_dim))
    def forward(self, x):
        return self.layers(x)

class ThreeLayerAllConnetedNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mid_layer = max(32, 2*in_dim)
        self.layers = nn.Sequential(
            nn.Linear(in_dim, self.mid_layer),
            nn.ReLU(),
            nn.Linear(self.mid_layer, self.mid_layer),
            nn.ReLU(),
            nn.Linear(self.mid_layer, out_dim))

    def forward(self, x):
        return self.layers(x)

class ThreeLayerConvolutionalNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, out_dim)

    def forward(self, x):
        x = functional.relu(self.conv1(x))
        x = functional.relu(self.conv2(x))
        x = functional.relu(self.conv3(x))
        x = functional.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)


# class CNNToDo(nn.Module): # TODO
#     def __init__(self, n_states, n_actions, mid_size=50):
#         super(Net, self).__init__()
#         # Create a DQN model using CUDA
#         self.device = torch.device("cuda" if False else "cpu")
#         self.layers = nn.Sequential(
#             nn.Linear(n_states, mid_size),
#             nn.ReLU(),
#             nn.Linear(mid_size, mid_size),
#             nn.ReLU(),
#             nn.Linear(mid_size, n_actions)
#         ).to(self.device)

#     def forward(self, x):
#         return self.layers(x)