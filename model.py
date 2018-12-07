# Actual DQN model
import numpy as np
import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class DQN(nn.Module):
    def __init__(self, out_size):
        '''
        :param out_size: number of actions in the game
        :input (N, C, H, W)
        :output (N)
        '''
        super().__init__()
        # Input (N, 4, 84, 84)
        self.layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), # (N, 32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # (N, 64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # (N, 64, 7, 7)
            nn.ReLU(),
            Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, out_size)
        )

    def forward(self, x):
        return self.layers(x)
