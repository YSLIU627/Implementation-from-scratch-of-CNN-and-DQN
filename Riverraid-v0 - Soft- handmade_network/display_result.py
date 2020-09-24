import torch
import gym
import random, math
import logging
from gym import wrappers
import atari_wrappers
from model import DQNModel
from dqn_agent import Agent
import time
import numpy as np

def test_result():
    #############
    #   test    #
    #############
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy_model = DQNModel(4,18)
    #policy_model.load_state_dict(torch.load('./data/dqn_Riverraid_qnetwork_target_state_dict.pt' ))
    #policy_model.eval()
    env = atari_wrappers.make_atari('RiverraidNoFrameskip-v4')
    env = atari_wrappers.wrap_deepmind(env, clip_rewards=True, frame_stack=True, pytorch_img=True)
    policy_model.load_model (torch.load('./data/dqn_Riverraid_qnetwork_target_state_dict.pickle'))
    num_episodes = 5
    episode = 1
    score = 0
    ep_score=[]
    done = False
    while (episode < num_episodes):
        observation = env.reset()
        done = False
        while not done :
            
        #action = agent.act(state)
            with torch.no_grad():
                t_observation /= 255
                #t_observation = t_observation.view(1, t_observation.shape[0], t_observation.shape[1], t_observation.shape[2])
                q_value = policy_model.forward(t_observation)
                action = argmax(q_value)
                env.render()
                time.sleep(0.0005)
                next_observation, reward, done, info = env.step(action)
                score += reward
                observation = next_observation

        if info['ale.lives'] == 0:
            episode += 1
            ep_score.append(score)
            score=0
    print ("Average Score : {}".format(int(np.mean(ep_score))))
    print(ep_score)

if (__name__=="__main__"):
    test_result()
