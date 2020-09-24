import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork

import os

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 2e-3            # no soft update
UPDATE_FREQUENCY = 1#5000#5000
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
In this game, no tensor is used

'''



class Agent():

    def __init__(self, state_size, action_size, seed):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed)
        self.qnetwork_target = QNetwork(state_size, action_size, seed)
        self.qnetwork_local.load_model("./dqn_LL_model data.pickle")
        self.qnetwork_target.load_model("./dqn_LL_model data.pickle")
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.loss =0
        self.loss_list =[]

    
    def step(self, state, action, reward, next_state, done, t_step):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = t_step 
        if self.t_step % UPDATE_EVERY == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > 100*BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
    
        """
        
        action_values = self.qnetwork_local.forward(state)
        

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values)
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            
        """
        states, actions, rewards, next_states, dones = experiences
        
        for time in range (BATCH_SIZE) :
            # compute Q_target from the target network inputing next_state
            Q_target_av = np.max(self.qnetwork_target.forward(next_states[time ]))
            Q_target = rewards[time ] + gamma*(Q_target_av)*(1-dones[time ]) # if done, than the second will not be added
            # compute the Q_expected 
            Q_expected = self.qnetwork_local.forward(states[time ]) # get q value for corrosponding action along dimension 1 of 64,4 matrix
            
            self.qnetwork_local.backward(Q_target, "MSE", actions[time])
            self.loss_list.append ((Q_target - Q_expected[actions[time]])**2) 
        self.loss =np.mean(self.loss_list)
        self.qnetwork_local.step()
        self.loss_list.clear()

        #  update target network #
        if self.t_step % UPDATE_FREQUENCY ==0:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = tau*θ_local + (1 - tau)*θ_target
        """
        self.qnetwork_target.soft_update(local_model, TAU)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """
        Randomly sample a batch of experiences from memory.
        """
        states, actions, rewards, next_states, dones = zip (*random.sample(self.memory,self.batch_size))
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return len(self.memory)