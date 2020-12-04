from __future__ import annotations
import torch
import torch.nn as nn

class TwoLayerAllConnetedNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()
        self.mid_layer = max(128, 2*in_dim)
        self.layers = nn.Sequential(
            nn.Linear(in_dim, self.mid_layer),
            nn.BatchNorm1d(self.mid_layer),
            nn.ReLU(),
            nn.Linear(self.mid_layer, out_dim))

    def forward(self, x):
        return self.layers(x)

class ThreeLayerAllConnetedNetwork(nn.Module):
    def __init__(self, n_states, n_actions, mid_size=50):
        super(Net, self).__init__()
        self.mid_layer = max(128, 2*in_dim)
        self.layers = nn.Sequential(
            nn.Linear(in_dim, self.mid_layer),
            nn.BatchNorm1d(self.mid_layer),
            nn.ReLU(),
            nn.Linear(self.mid_layer, self.mid_layer),
            nn.BatchNorm1d(self.mid_layer),
            nn.ReLU(),
            nn.Linear(self.mid_layer, out_dim))

    def forward(self, x):
        return self.layers(x)


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