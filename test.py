import gym
from kaggle_environments import make
import numpy as np

import torch
from double_dqn_agent import DQNAgent


import matplotlib.pyplot as plt
from collections import deque

# this is a test script which runs training with a Double DQN Agent

class OBSParser(object):

    @staticmethod
    def parse(obs):
        # parse left players units
        l_units = [[x[0] for x in obs['left_team']], [x[1] for x in obs['left_team']],
                   [x[0] for x in obs['left_team_direction']], [x[1] for x in obs['left_team_direction']],
                   obs['left_team_tired_factor'], obs['left_team_yellow_card'],
                   obs['left_team_active'], obs['left_team_roles']
                  ]

        l_units = np.r_[l_units].T

        # parse right players units
        r_units = [[x[0] for x in obs['right_team']], [x[1] for x in obs['right_team']],
                   [x[0] for x in obs['right_team_direction']], [x[1] for x in obs['right_team_direction']],
                   obs['right_team_tired_factor'],
                   obs['right_team_yellow_card'],
                   obs['right_team_active'], obs['right_team_roles']
                  ]

        r_units = np.r_[r_units].T
        # combine left and right players units
        units = np.r_[l_units, r_units].astype(np.float32)

        # get other information
        game_mode = [0 for _ in range(7)]
        game_mode[obs['game_mode']] = 1
        scalars = [*obs['ball'],
                   *obs['ball_direction'],
                   *obs['ball_rotation'],
                   obs['ball_owned_team'],
                   obs['ball_owned_player'],
                   *obs['score'],
                   obs['steps_left'],
                   *game_mode,
                   *obs['sticky_actions']]

        scalars = np.r_[scalars].astype(np.float32)
        # get the actual scores and compute a reward
        l_score,r_score = obs['score']
        reward = l_score - r_score
        reward_info = l_score,r_score,reward
        return (units[np.newaxis, :], scalars[np.newaxis, :]),reward_info


class FootEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_id=0):
        super(FootEnv, self).__init__()
        self.env_id = env_id
        self.agents = [None, 'run_right']
        self.env = make("football", configuration={"save_video": False,
                                                   "scenario_name": "11_vs_11_kaggle",
                                                   "running_in_notebook": False})
        self.trainer = None


    def step(self, action):
        obs, reward, done, info = self.trainer.step([action])
        obs = obs['players_raw'][0]
        state,(l_score,r_score,custom_reward) = OBSParser.parse(obs)
        info['l_score'] = l_score
        info['r_score'] = r_score
        return state, custom_reward, done, info

    def reset(self):
        self.trainer = self.env.train(self.agents)
        obs = self.trainer.reset()
        obs = obs['players_raw'][0]
        state,_ = OBSParser.parse(obs)
        return state

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        pass

def env_fn(env_id=1):
    return FootEnv(env_id=env_id)


agent = DQNAgent(state_size=207,action_size=19,seed=0)

def dqn(tag, agent=agent,n_episodes=5000,max_t=1000,eps_start=1.0,eps_end=0.01,eps_decay=0.995):
    env = env_fn() 
    
    scores=[]
    scores_window = deque(maxlen=100)
    eps = eps_start

    for episode in range(1,n_episodes+1):
        state = env.reset()
        score = 0
        best_score = 0
        for i in range(max_t):
            # choose an action given a state by using our agent
            action = int(agent.act(state,eps)) 
            # execute action in environment and obtain next_state, reward, done flag and other information
            next_state, reward, done, _ = env.step(action)
            # update score
            score+=reward
            # add the experience to replay buffer - if replay buffer is filled, it will start learning
            agent.step(state,action,reward,next_state,done)
            # update state for next time step
            state = next_state
            if done:
                break
        # add score to deque
        scores_window.append(score)
        # add score to per episode score list
        scores.append(score)
        # reduce epsilon by decay constant
        eps = max(eps_end,eps_decay*eps)

        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(episode,np.mean(scores_window),score),end="")
        if episode%100==0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(episode,np.mean(scores_window),score))
        if score > best_score:
            print('\rScore of {:.2f} beats previous best score of {:.2f}, saving model weights'.format(score,best_score))
            #save high scoring model weights
            torch.save(agent.local_net.state_dict(),'{}-test-highscore.pth'.format(tag))
            best_score = score
        
        # save final weights as well
        torch.save(agent.local_net.state_dict(),'{}-finalweights.pth'.format(tag))

    return scores
    
dqn_scores = dqn('double_dqn')

fig = plt.figure()
plt.plot(np.arange(len(dqn_scores)),dqn_scores)
plt.ylabel('Reward')
plt.xlabel('Episode #')
plt.show()
    
    
