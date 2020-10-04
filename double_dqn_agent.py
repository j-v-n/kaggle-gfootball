import numpy as np
import random
from collections import deque, namedtuple
import gym

from dqn_model import SoccerNet


import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

# HYPER-PARAMETERS CHOSEN BY ROUGH EXPERIMENTATION
BATCH_SIZE = 128
REPLAY_SIZE = int(1e5)
GAMMA = 0.99
TAU = 0.005
LR = 1e-4
UPDATE_EVERY = 2

# I work locally with a GPU, therefore the agent always uses CUDA
device = torch.device("cuda:0")


class DQNAgent:

    """ 
    Class to handle DQN Agent
    
    """

    def __init__(self, state_size, action_size, seed):

        self.seed = random.seed(seed)

        self.state_size = state_size
        self.action_size = action_size

        self.local_net = SoccerNet(state_size, action_size).to(device) #creating local network
        self.target_net = SoccerNet(state_size, action_size).to(device) #creating target network

        self.optimizer = optim.Adam(self.local_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(action_size, REPLAY_SIZE, BATCH_SIZE, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        ''' 
        Defining the step method which adds a state-action-reward-next_state-done tuple to the replay buffer.
        If the buffer has reached a target size, learning commences
        '''
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) % UPDATE_EVERY == 0:
            if len(self.memory) > 2 * BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        ''' 
        Defining the learn method using the Double DQN method
        '''
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            next_state_acts = self.local_net(next_states)  
            next_state_acts_max = next_state_acts.max(1)
            next_state_acts_max1 = next_state_acts_max[1]
            next_state_acts_uns = next_state_acts_max1.unsqueeze(1)
            Q_next = self.target_net(next_states).gather(1, next_state_acts_uns)
            Q_targets = rewards + gamma * Q_next * (1 - dones)

        Q_local = self.local_net(states).gather(1, actions)

        loss = F.mse_loss(Q_local, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.local_net, self.target_net, TAU)

    def soft_update(self, local_model, target_model, tau):
        ''' 
        Defining a soft update method which updates the parameters of the target network
        by annealing the parameters of the local network
        '''
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def act(self, state, eps=0):
        '''
        Defining the act method which selects an action given a state
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.local_net.eval()
        with torch.no_grad():
            action_values = self.local_net(state)
        self.local_net.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))


class ReplayBuffer:

    """
    Class to handle replay buffer. 
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)

        self.batch_size = batch_size

        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        '''
        Defining the add method which adds state-action-reward-next_state-done to the replay buffer
        '''
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        '''
        Defining the sample method which samples a batch of states, actions, rewards, next_states and dones
        from the replay buffer
        '''
        experiences = random.sample(self.memory, k=self.batch_size)

        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .long()
            .to(device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(
                np.vstack([e.next_state for e in experiences if e is not None])
            )
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(
                (np.vstack([e.done for e in experiences if e is not None])).astype(
                    np.uint8
                )
            )
            .float()
            .to(device)
        )

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

