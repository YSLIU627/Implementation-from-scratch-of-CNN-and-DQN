
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from collections import deque

from dqn_agent import Agent
import time
from plot_uti import plot_score
import pickle
###########
# hyperparameter #
Episodes=4000
Max_t=1000
Eps_start=1.0
Eps_end=0.1
Eps_decay=0.995
Benchmark=200

def game_start(game_name):
    '''
    initial the environment
    '''
    env = gym.make(game_name)
    env.seed(0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)

    agent = Agent(state_size=8, action_size=4, seed=0)
    return env,agent
    

def dqn(env,agent,episodes=Episodes, max_t=Max_t, eps_start=Eps_start, eps_end=Eps_end, eps_decay=Eps_decay):
    
    ten_epi_scores = []
    ten_epi_loss = []
    loss_every_ten_epi = []
    scores_every_ten_epi=[]   
                       
    eps = eps_start                    # initialize epsilon
    t_step = 0 
    
    for i_episode in range(1, episodes+1):
        state = env.reset()
        score = 0
        
        ##### training process   #####
        for t in range(max_t):
            # import the agent
            action = agent.act(state, eps)
            #env.render()
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done, t_step)
            state = next_state
            score += reward
            t_step += 1
            if done:
                break 
        ten_epi_scores.append(score)             
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        ten_epi_loss.append(agent.loss)
        if i_episode % 100 == 0:
            
            pickle.dump(agent.qnetwork_target.data,open ('./dqn_LL_model data.pickle','wb'))
            pickle.dump(scores_every_ten_epi, open('./dqn_LunarLander_scores.pickle', 'wb'))
            pickle.dump(loss_every_ten_epi, open('./dqn_LunarLander_loss.pickle', 'wb'))

        if i_episode % 10 == 0:
            scores_every_ten_epi.append([ t_step , np.mean (ten_epi_scores)])
            print('\rEpisode {}\tAverage Score: {:.2f}\t Loss:{:.2f}'.format(i_episode, np.mean (ten_epi_scores), np.mean(ten_epi_loss)))
            ten_epi_scores.clear()
            loss_every_ten_epi.append([t_step, np.mean(ten_epi_loss)])
            ten_epi_scores.clear()
            
        
    



def Display_Result(env,agent):
    # show the result
    # load the weights from file
    agent.qnetwork_local.load_model('./dqn_LL_model data.pickle')   

    for i in range(3):
        state = env.reset()
        for j in range(1000):
            action = agent.act(state)
            env.render()
            time.sleep(0.0005)
            state, reward, done, _ = env.step(action)
            if done:
                break 
                
    env.close()

if __name__ == '__main__':
    env,agent=game_start('LunarLander-v2')
    dqn(env,agent,episodes=Episodes, max_t=Max_t, eps_start=Eps_start, eps_end=Eps_end, eps_decay=Eps_decay)
    plot_score()
    Display_Result(env,agent)
    print("Success")


