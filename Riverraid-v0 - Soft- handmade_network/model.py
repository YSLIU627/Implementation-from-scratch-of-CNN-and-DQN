from cnn import ConvLayer
from FCLayer import FCLayer 
learning_rate= 0.00015
BATCH_SIZE=16
import pickle
import numpy as np
import torch
import torch.nn.functional as F
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
class DQNModel():

    def __init__(self, in_channels, action_size):
    
        super(DQNModel, self).__init__()   # the line you give simply calls the __init__ method of ClassNames parent class.
        self.dict = {}                     # contain all the parameters
        self.conv1 = ConvLayer(in_channels,32,84,8,4,"relu",learning_rate,BATCH_SIZE,self.dict,"conv1")
        self.conv2 = ConvLayer(32,64,self.conv1.output_size,4,2,"relu",learning_rate,BATCH_SIZE,self.dict,"conv2") 
        self.conv3=  ConvLayer(64,64,self.conv2.output_size,3,1,"relu",learning_rate,BATCH_SIZE,self.dict,"conv3")
        self.fc1 = FCLayer(64*self.conv3.output_size**2 , 512,"relu",learning_rate,BATCH_SIZE,self.dict,"fc1")
        self.out = FCLayer(512, action_size,"identity" ,learning_rate,BATCH_SIZE,self.dict,"out")
        self.action_size =action_size
        self.in_channels =in_channels
        
        
        
    def forward(self, state):
        self.conv1.forward(state)
        self.conv2.forward(self.conv1.output_tensor)
        self.conv3.forward(self.conv2.output_tensor)
        self.fc1.forward(self.conv3.output_tensor.reshape(-1,64*self.conv3.output_size**2))
        self.out.forward(self.fc1.output_tensor_act)

        return self.out.output_tensor_act
    def edit_tensor(self, tensor):
        '''
        reshape the delta tensor, transforming  (B,H) to (B,C,H,W)
        '''
        return tensor.reshape (BATCH_SIZE,-1,self.conv3.output_size,self.conv3.output_size)
    def backward(self,target,local, loss_name,actions) :
        
        self.out.backward(self.out.input_tensor, self.loss_gradient(self.out.output_tensor_act, target, local,loss_name,actions))
        self.fc1.backward(self.fc1.input_tensor, self.out.delta_tensor)
        self.conv3.backward(self.conv3.input_tensor, self.edit_tensor(self.fc1.delta_tensor),"relu")  
        self.conv2.backward(self.conv2.input_tensor, self.conv3.delta_tensor,"relu")
        self.conv1.backward(self.conv1.input_tensor, self.conv2.delta_tensor,"relu")

    def step (self):
        self.conv1.update()
        self.conv2.update()
        self.conv3.update()
        self.fc1.update()
        self.out.update()
    

    
    def loss_gradient(self, tensor ,target, local, loss_name, actions):
        '''
        calculate the loss that depends on the defintion of the loss

        '''
        g = torch.zeros_like(tensor,device=device)
        if loss_name =="huber":
            n = torch.abs(local - target)
            loss_g = torch.where(n < 1, local - target, n**0)
            loss_g = torch.where(local - target <-1 , - n**0,loss_g )
            for b in range(BATCH_SIZE):
                action = actions[b,0]
                g[b,action.int()] = loss_g[b,0]
            return g
        raise 

    def soft_update (self, local_network,TAU):
        '''
        a method for updating in soft way
        
        '''
        self.conv1.filters.weights = TAU *local_network.conv1.filters.weights + (1-TAU) *self.conv1.filters.weights
        self.conv1.filters.bias =    TAU *local_network.conv1.filters.bias +    (1-TAU) *self.conv1.filters.bias
        self.conv2.filters.weights = TAU *local_network.conv2.filters.weights + (1-TAU) *self.conv2.filters.weights
        self.conv2.filters.bias =    TAU *local_network.conv2.filters.bias +    (1-TAU) *self.conv2.filters.bias
        self.conv3.filters.weights = TAU *local_network.conv3.filters.weights + (1-TAU) *self.conv3.filters.weights
        self.conv3.filters.bias =    TAU *local_network.conv3.filters.bias +    (1-TAU) *self.conv3.filters.bias
        self.fc1.filter.weights =    TAU *local_network.fc1.filter.weights +    (1-TAU) *self.fc1.filter.weights
        self.fc1.filter.bias =       TAU *local_network.fc1.filter.bias +       (1-TAU) *self.fc1.filter.bias
        self.out.filter.weights =    TAU *local_network.out.filter.weights +    (1-TAU) *self.out.filter.weights
        self.out.filter.bias =       TAU *local_network.out.filter.bias +       (1-TAU) *self.out.filter.bias
        
    def get_dict(self):
        '''
        return the dictionary of all the learned parameters of the model
        '''
        return self.dict   
        
    def load_model(self, dict):
        '''
        a method to upload the dict of previous model
        '''
        self.dict.update(dict)
    




    