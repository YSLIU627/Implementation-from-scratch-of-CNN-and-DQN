import numpy as torch
from my_optimzer import Adam
import numpy as torch
import torch
import torch.nn.functional as F
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class Filter():
    '''
    weight and bias class for each layer
    '''
    def __init__(self, height,width ,BATCH_SIZE,learning_rate,dict,name):
        self.BATCH_SIZE =BATCH_SIZE
        self.learning_rate =learning_rate
        self.name = name
        self.dict = dict
        self.dict[name+"_w"] = torch.randn(( height , width),device=device).double() / (width)
        self.dict[name+"_b"] = torch.zeros(height,device=device).double()
        self.weights = dict[name+"_w"]
        self.bias    = dict[name+"_b"]
        self.weights_grad = torch.zeros(self.weights.shape,device=device).unsqueeze(0).repeat(self.BATCH_SIZE,1,1)
        self.bias_grad = torch.zeros(self.bias.shape,device=device).unsqueeze(0).repeat(self.BATCH_SIZE,1)
        self.adam_w = Adam(self.weights.shape,self.learning_rate)
        self.adam_b = Adam(self.bias.shape,self.learning_rate)

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias


###### here we will step the filter (Handmade Adam Optimizer)
    def step(self, learning_rate):
        self.weights -= self.adam_w.update(torch.sum(self.weights_grad,0)/self.BATCH_SIZE).to(device)
        self.bias -= self.adam_b.update(torch.sum(self.bias_grad,0)/self.BATCH_SIZE).to(device)
        


class FCLayer():
    '''
    basic class for each layer
    '''
    def __init__(self, input_length, 
                 output_length,  
                 activator,
                 learning_rate,BATCH_SIZE,dict,name):
        self.BATCH_SIZE =BATCH_SIZE
        self.learning_rate = learning_rate
        self.dict =dict
        self.name =name
        self.input_width = input_length
        self.output_length = output_length
        self.filter_width = input_length
        self.filter_height = output_length
        self.output_width = output_length
        self.filter = Filter(self.filter_height,self.filter_width,self.BATCH_SIZE,self.learning_rate,self.dict,self.name)
        self.delta_tensor = None
        self.activator = activator
        self.parameters = [self.filter.weights, self.filter.bias]

    def forward(self, input_tensor):
        '''
        calculate the ouput
        and save the result on the self.output_tensor
        '''
        self.input_tensor = input_tensor
        self.output_tensor = F.linear(self.input_tensor,self.filter.weights,bias=self.filter.bias)
        x = self.output_tensor
        if self.activator == "relu":
            self.output_tensor_act = torch.where(x>0, x, torch.zeros_like(x))
            return
        if self.activator == "identity":
            self.output_tensor_act =x
            return
        raise 
    
    
    

    def backward(self, input_tensor, sensitivity_tensor):
        '''
        backward process
        '''
        self.dz_calculate(sensitivity_tensor)                        
        self.linear_backward()
        
    def dz_calculate(self, sensitivity_tensor):
        self.dZ= sensitivity_tensor
        '''
        calculate dZ(Z: the output before activation)
        '''
        if self.activator == "relu":
            x = self.output_tensor
            self.dZ = torch.where(x>0,self.dZ,torch.zeros_like(x))
            
            return 
        if self.activator == "identity":
            return
        raise


    def linear_backward(self):
        '''
        Z=WA+b
	    param dZ: Upstream derivative, the shape (n^[l+1],m)
	    param A: input of this layer 
	    
	    '''
        dZ= self.dZ
        A, W, b = self.input_tensor, self.filter.weights, self.filter.bias
        self.filter.weights_grad = torch.matmul(dZ.reshape(self.BATCH_SIZE,-1,1), A.reshape(self.BATCH_SIZE,1,-1))
    
        self.filter.bias_grad = dZ
        
        self.delta_tensor = F.linear(dZ, W.transpose(0,1))

    def update(self):
        '''
        update the filter
        '''
        self.filter.step(self.learning_rate)


    