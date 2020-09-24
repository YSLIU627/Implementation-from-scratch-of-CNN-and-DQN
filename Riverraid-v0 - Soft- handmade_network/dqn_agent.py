import torch
import logging
import pickle
from collections import deque
import gym
from gym import wrappers
import atari_wrappers
import numpy as np
import random
from collections import namedtuple, deque
from model import DQNModel
import torch
import torch.nn.functional as F
import torch.nn
import os
BATCH_SIZE = 16
#target_update = 5000 # for no soft update algorithm
gamma = 0.95
BUFFER_SIZE = 250000
skip_frame = 4   # how often to update the network
LR = 0.00025
Alpha = 0.95
TAU = 2e-3  # no soft update
epsilon_start = 1.0
epsilon_final = 0.1
epsilon_decay = 200000#400000
UPDATE_FREQUENCY = 1 # soft-update

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, in_channels, action_size, seed):
        """Initialize an Agent object.
        """
        self.in_channels = in_channels
        self.action_size = action_size
        #self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = DQNModel(in_channels, action_size)
        self.qnetwork_target = DQNModel(in_channels, action_size)
    
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        self.loss_list = []
    
    def step(self, observation, action, reward, next_observation, done,num_frames):
        # Save experience in replay memory
        self.memory.add(observation, action, reward, next_observation, done)
        self.t_step = num_frames
        # Learn every UPDATE_EVERY time steps.
        if self.t_step %  skip_frame== 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                #experiences = self.memory.sample()
                self.learn()

    def act(self, observation, eps=0.):
        #Returns actions for given observation as per current policy.
        t_observation = torch.from_numpy(observation).double()/255  
        # gray standard
        t_observation = t_observation.unsqueeze(0).to(device)
        

        # Epsilon-greedy action selection
        if random.random() > eps:
            action_values = self.qnetwork_local.forward(t_observation)
            action = action_values.argmax(1).data.cpu().numpy().astype(int)[0]
            # note the d of argmax , if the tensor is 4d then the para of argmax should be 2
        else:
            action = random.sample(range(self.action_size), 1)[0]
        return action

    def learn(self):
        
        observations, actions, rewards, next_observations, dones =  self.memory.sample()
        
        observations = torch.from_numpy(np.array(observations) / 255).double().to(device)
            
        actions = torch.from_numpy(np.array(actions).astype(int)).int().to(device)
        actions = actions.view(actions.shape[0], 1)
            
        rewards = torch.from_numpy(np.array(rewards)).double().to(device)
        rewards = rewards.view(rewards.shape[0], 1)
            
        next_observations = torch.from_numpy(np.array(next_observations) / 255).double().to(device)
        
        dones = torch.from_numpy(np.array(dones).astype(int)).int().to(device)
        dones = dones.view(dones.shape[0], 1)
        
        Q_target_next = self.qnetwork_target.forward(next_observations).max(1)[0].unsqueeze(1)
        Q_target = rewards + gamma*(Q_target_next)*(1-dones) # if done, than the second will not be added
        # compute the Q_local 
        Q_local = self.qnetwork_local.forward(observations).gather(1, actions.long())
        loss = self.huber_loss(Q_local, Q_target)
        self.qnetwork_local.backward(Q_target,Q_local, "huber",actions)
        self.loss_list.append(loss.cpu().numpy())
        self.qnetwork_local.step()
        #  update target network #
        if self.t_step % UPDATE_FREQUENCY == 0:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = tau*θ_local + (1 - tau)*θ_target
        """
        self.qnetwork_target.soft_update(local_model, TAU)
     
    
    
    def huber_loss(self, input, target, beta=1, size_average=True):
        """
        a method of  defining loss which increase the robustness of computing on discrete data
        """
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        if size_average:
            return loss.mean()
        return loss.sum()

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        #self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.memory.append([state, action, reward, next_state, done])
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        observation, action, reward, next_observation, done = zip (*random.sample(self.memory,self.batch_size))
        
        return observation, action, reward, next_observation, done

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)



    