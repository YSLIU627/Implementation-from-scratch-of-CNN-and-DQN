import random, math
from collections import deque
import matplotlib.pyplot as plt
import pickle
import logging
import time
import numpy as np
import gym
from gym import wrappers
import atari_wrappers
from model import DQNModel
from dqn_agent import Agent
from plot_uti import plot_score
from display_result import test_result
import torch
frames_max = 1000000
BUFFER_INI = 4#0000#40000
#batch_size = 16
#target_update = 50#5000
#GAMMA = 0.99
#buffer_size = 250000
#skip_frame = 4
#learning rate = 0.00030
#TAU = 2e-3
epsilon_start = 1.0
epsilon_final = 0.1
epsilon_decay = 200000

def initialize_env():

    env = atari_wrappers.make_atari('RiverraidNoFrameskip-v4')
    env = atari_wrappers.wrap_deepmind(env, clip_rewards=False, frame_stack=True, pytorch_img=True)
    agent = Agent(in_channels=4, action_size=18, seed=0)
    
    ####initial network####
    agent.qnetwork_target.load_model (torch.load('./data/dqn_Riverraid_qnetwork_target_state_dict.pth'))
    agent.qnetwork_local.load_model  (torch.load( './data/dqn_Riverraid_local_model_state_dict.pth'))
    
    ####initial the buffer replay####
    while len(agent.memory) < BUFFER_INI:
        observation = env.reset()
        done = False
        while not done:
            action = random.sample(range(env.action_space.n), 1)[0]    
            next_observation, reward, done, info = env.step(action)    
            agent.memory.add(observation, action, reward, next_observation, done)
            observation = next_observation
    print( "Replay Buffer Initialized")
    return env,agent

#### define the decretion of the epsilon ####
epsilon_by_frame = lambda step_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * step_idx / epsilon_decay)

def DQN(env,agent,frames_max):
    
    #### logging ####
    logger = logging.getLogger('dqn_Riverraid-v0')
    logger.setLevel(logging.INFO)
    logger_handler = logging.FileHandler('./data/dqn_Riverraid-v0.log')
    logger.addHandler(logger_handler)
    
    scores_every_five_epi = []            # list containing scores from each episode
    five_epi_scores = []
    loss = 0
    five_epi_loss = []
    eps = epsilon_start                    # initialize epsilon
    num_frames=0
    score=0
    episode=1
    #### training process ####
    while num_frames < frames_max :   
        observation = env.reset()
        done = False
        while not done :
            # import the agent
            action = agent.act(observation, eps)
            #env.render()
            next_observation, reward, done, info = env.step(action)  #action
            score += reward
            num_frames += 1
            agent.step(observation, action, reward, next_observation, done,num_frames)  #update
            observation = next_observation
            eps=epsilon_by_frame(num_frames)

        # if all the agent runs out all the lives then an episode is done  
        if info['ale.lives'] == 0:
            five_epi_scores.append(score)
            score = 0
            episode += 1

            
            if episode % 20 == 0:
                logger.info('Frame: ' + str(num_frames) + ' / Episode: ' + str(episode) + ' / Epsilon ' + str(eps) + ' / Average Score (over last 5 episodes): ' + str(int(np.mean(five_epi_scores))))
                pickle.dump(scores_every_five_epi, open('./data/dqn_Riverraid_mean_scores.pickle', 'wb'))
            
            if(episode % 5 == 0):
                scores_every_five_epi.append ([num_frames , np.mean(five_epi_scores)])
                five_epi_scores.clear()
                loss = np.mean(agent.loss_list)
                five_epi_loss.append(loss)
                agent.loss_list.clear()
                print("Frames:{}  \t Loss:{:.2f}".format(num_frames, loss))
            

            if episode % 100 == 0:
                torch.save(agent.qnetwork_local.get_dict(),'./data/dqn_Riverraid_local_model_state_dict.pth')
                torch.save(agent.qnetwork_target.get_dict(),'./data/dqn_Riverraid_qnetwork_target_state_dict.pth')
        
    
if(__name__== "__main__"):
     
    env,agent=initialize_env()
    DQN(env,agent,frames_max)
    plot_score()
    print("The next is testing time")
    test_result()
    print("Finished")



