
import torch.nn.functional as F

from FCLayer import FCLayer 
learning_rate= 5e-4 
import pickle
from activators import ReluActivator, IdentityActivator
import numpy as np

class QNetwork():

    def __init__(self, state_size, action_size, seed):
        
        super(QNetwork, self).__init__()   # the line you give simply calls the __init__ method of ClassNames parent class.
        self.seed = seed.random(seed)
        self.fc1 = FCLayer(state_size, 256,ReluActivator(),learning_rate)
        self.fc2 = FCLayer(256,128,ReluActivator(),learning_rate)
        self.fc3 = FCLayer(128,64,ReluActivator(),learning_rate)
        self.out = FCLayer(64, action_size,IdentityActivator() ,learning_rate)
        self.data = [self.fc1.parameters,self.fc2.parameters,self.fc3.parameters,self.out.parameters]
        
        
    def forward(self, state):
        self.fc1.forward(state.reshape(-1,1))
        self.fc2.forward(self.fc1.output_array_act)
        self.fc3.forward(self.fc2.output_array_act)
        self.out.forward(self.fc3.output_array_act)
        return self.out.output_array_act  

    def backward(self,target, loss_name ,action)  :
        
        self.out.backward(self.out.output_array, self.loss_gradient(self.out.output_array_act, target, loss_name, action))
        self.fc3.backward(self.fc3.output_array, self.out.delta_array)
        self.fc2.backward(self.fc2.output_array, self.fc3.delta_array)
        self.fc1.backward(self.fc1.output_array, self.fc2.delta_array)
        

    def step (self):
        '''
        update the network
        '''
        self.fc1.update()
        self.fc2.update()
        self.fc3.update()
        self.out.update()

    
    def loss_gradient(self, local ,target, loss_name, action):
    '''
    the loss_gradient depends on the defintion of the loss

    '''   
        ABS =abs (target-local[action])
        grad = np.zeros(local.shape)
        
        if loss_name == "MSE":
            grad[action][0] += 2* (local[action][0] - target)
            return grad
        raise 

    def soft_update (self, local_network,TAU):
        '''
        a method for updating in soft way
        '''

        self.fc1.filter.weights = TAU* local_network.fc1.filter.weights + (1-TAU) *self.fc1.filter.weights
        self.fc1.filter.bias = TAU* local_network.fc1.filter.bias + (1-TAU) *self.fc1.filter.bias
        self.fc2.filter.weights = TAU* local_network.fc2.filter.weights + (1-TAU) *self.fc2.filter.weights
        self.fc2.filter.bias = TAU* local_network.fc2.filter.bias +(1-TAU) * self.fc2.filter.bias
        self.fc3.filter.weights = TAU* local_network.fc3.filter.weights +(1-TAU) * self.fc3.filter.weights
        self.fc3.filter.bias = TAU* local_network.fc3.filter.bias + (1-TAU) *self.fc3.filter.bias
        self.out.filter.weights = TAU* local_network.out.filter.weights + (1-TAU) *self.out.filter.weights
        self.out.filter.bias = TAU* local_network.out.filter.bias + (1-TAU) *self.out.filter.bias
        

    def load_model(self, path):
        '''
        a method to upload the dict of previous model
        '''
        self.data = pickle.load(open(path,'rb'))
