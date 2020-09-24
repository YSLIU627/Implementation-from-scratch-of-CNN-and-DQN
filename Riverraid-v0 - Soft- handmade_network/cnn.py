import numpy as np
import torch.nn.functional as F
import torch
from my_optimzer import Adam

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
class Filter(object):
    '''
    the filter used by convolution
    '''
    def __init__(self,channel_out, channel_in, height ,width,learning_rate, BATCH_SIZE  ,dict,name):
        self.learning_rate = learning_rate
        self.BATCH_SIZE =BATCH_SIZE
        self.dict =dict
        self.name = name
        self.dict[name+"_w"] = torch.randn((channel_out, channel_in, height, width)).double()/width
        self.dict[name+"_b"] = torch.zeros( channel_out,device=device)
        self.weights = self.dict[name+"_w"]
        self.bias = self.dict[name+"_b"]
        self.weights_grad = torch.zeros_like(self.weights,device=device).unsqueeze(0).repeat(BATCH_SIZE,1,1,1,1)
        self.bias_grad = torch.zeros((BATCH_SIZE, channel_out),device=device)
        self.channel_out = channel_out
        self.optim_w = Adam(self.weights.shape ,self.learning_rate)
        self.optim_b = Adam(self.bias.shape,self.learning_rate)

    def get_weights(self):
        return self.weights
    def get_bias(self):
        return self.bias

    def update(self, learning_rate):
        self.weights -= self.optim_w.update (torch.sum(self.weights_grad, 0)/self.BATCH_SIZE).to(device)
        self.bias -= self.optim_b.update (torch.sum(self.bias_grad,0)/self.BATCH_SIZE).to(device)

class ConvLayer(object):
    '''
    basic class of conv layer
    '''
    def __init__(self,channel_in , channel_out, input_size,filter_size , stride, activator,
                 learning_rate , BATCH_SIZE,dict,name):
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.channel_in = channel_in
        self.dict = dict
        self.name = name
        self.filter_size = filter_size
        self.filter_size = filter_size
        self.channel_out = channel_out
        self.zero_padding = 0
        self.stride = stride
        self.BATCH_SIZE = BATCH_SIZE
        self.output_width = int(\
            ConvLayer.calculate_output_size(
            self.input_size, filter_size, self.zero_padding,
            stride))
        self.output_height = int(\
            ConvLayer.calculate_output_size(
            self.input_size, filter_size, self.zero_padding,
            stride))
        self.output_size = self.output_height
        self.filters = Filter(self.channel_out, self.channel_in, self.filter_size,self.filter_size,
                        self.learning_rate,self.BATCH_SIZE ,self.dict, self.name)
        self.activator = activator
        
        self.parameters= self.filters.parameters
        self.expanded_tensor= None
        self.zp=0

    def forward(self, input_tensor):
        '''
        forward input, save the result on the self.output_tensor
        '''
        self.input_tensor = input_tensor.double()
        self.output_tensor = F.conv2d(input_tensor, self.filters.get_weights(),self.filters.get_bias(),stride=self.stride)
        if self.activator == "relu":
            self.output_tensor=torch.where(self.output_tensor>0,self.output_tensor,torch.zeros_like(self.output_tensor))
            return self.output_tensor
        if self.activator == "identity":
            return self.output_tensor
        
        raise
    def create_delta_tensor(self):
        return torch.zeros((self.BATCH_SIZE, self.channel_in,self.input_size, self.input_size),device=device)

    def backward(self, input_tensor, sensitivity_tensor, activator):
        '''
        backward process and save the gradients on the weights_grad
        '''
        self.bp_sensitivity_map(sensitivity_tensor,activator)
        self.bp_gradient(sensitivity_tensor)

    def update(self):
        '''
        update the parameters for each filter
        '''
        self.filters.update(self.learning_rate)

    def bp_sensitivity_map(self, sensitivity_tensor,activator):
        '''
        Calculate by the sensitive tensor of this layer,
        save the frontier delta tensor on the self.delta_tensor
        
        '''
        # expand the original sensitivity map
        self.expanded_tensor = self.expand_sensitivity_map(sensitivity_tensor).double()
        # fullconvolution ,apply  sensitivitiy map with zero padding
        # TIP : No need to calculate the zero-padding part of original input
        expanded_width = self.expanded_tensor.shape[3]
        self.zp = (self.input_size +  self.filter_size - 1 - expanded_width) / 2
        self.delta_tensor = self.create_delta_tensor()
        
        
        # calculate the delta_tensor for each filter, and add them up       
        flipped_weights = torch.rot90(self.filters.get_weights(), 2,[2,3]).reshape(self.channel_in,self.channel_out,self.filter_size,self.filter_size)
            
        self.delta_tensor = F.conv2d(self.expanded_tensor, flipped_weights,padding=int(self.zp))
        
        if self.activator == "relu":
            self.delta_tensor= torch.where(self.input_tensor >0,self.delta_tensor,torch.zeros_like(self.input_tensor)) 
            
            return self.delta_tensor
        if self.activator == "identity":
            #self.delta_tensor *= derivative_tensor
            return self.delta_tensor
        raise

    def bp_gradient(self, sensitivity_tensor):
        '''
        get the gradient of weights and bias through back propagation 
        '''
        for b in range(self.BATCH_SIZE):
            grad = (F.conv2d(self.input_tensor[b,:,:,:].unsqueeze(1),
                    self.expanded_tensor[b,:,:,:].unsqueeze(1))).reshape(self.channel_out,-1,self.filter_size,self.filter_size)
            
        self.filters.bias_grad = torch.sum(self.expanded_tensor[:,:,:,:],(2,3))



    def expand_sensitivity_map(self, sensitivity_tensor):
        
        channel_out =self.channel_out
        # When stride =1 , calculate the size of the sensitivity map
        expanded_width = (self.input_size - 
            self.filter_size + 2 * self.zero_padding + 1)
        expanded_height = (self.input_size - 
            self.filter_size + 2 * self.zero_padding + 1)
        # construct new sensitivity_map
        expand_tensor = torch.zeros((self.BATCH_SIZE,channel_out, expanded_height, expanded_width),device=device)
        # copy loss from original sensitivity map
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_tensor[:,:,i_pos,j_pos] = sensitivity_tensor[:,:,i,j]
        return expand_tensor


    @staticmethod
    def calculate_output_size(input_size,
            filter_size, zero_padding, stride):
        '''
        calculate the output size
        '''
        return (input_size - filter_size + 
            2 * zero_padding) / stride + 1

