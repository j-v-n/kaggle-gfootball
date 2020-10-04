import torch
import torch.nn as nn
import numpy as np

class SoccerNet(nn.Module):

    def __init__(self,input_shape,n_actions):

        super(SoccerNet,self).__init__()

        self.fc= nn.Sequential(
            nn.Linear(input_shape,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,n_actions)
        )

    def forward(self,state):
        return self.fc(state)