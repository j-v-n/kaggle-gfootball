from kaggle_environments.envs.football.helpers import *
import torch
from double_dqn_agent import DQNAgent




def agent_sub(obs):
    model = DQNAgent(state_size=115,action_size=19,seed=0)

    model.local_net.load_state_dict(torch.load('double-dqn-test.pth'))

    model.local_net.eval()
    
    obs = obs['players_raw'][0]
    action = model.act(obs)

    return action
