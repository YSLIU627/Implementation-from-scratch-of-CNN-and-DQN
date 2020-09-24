import numpy as np
from activators import ReluActivator, IdentityActivator
from my_optimzer import Adam
BATCH_SIZE = 64


class Filter():
    '''
    weight and bias class for each layer
    '''
    def __init__(self, height,width):
        
        self.weights = np.random.randn( height , width) / np.sqrt(width)
        self.bias = np.zeros(( height , 1))
        self.weights_grad_batch =[ ]
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = np.zeros(self.bias.shape)
        self.bias_grad_batch = []
        self.adam_w = Adam(self.weights.shape)
        self.adam_b = Adam(self.bias.shape)

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias



    def step(self, learning_rate):
        '''
        update the weights and bias according to the given optimizer
        '''
        weights_grad = np.zeros(self.weights.shape)
        bias_grad = np.zeros(self.bias.shape)
        for i in range (BATCH_SIZE):
            weights_grad += self.weights_grad_batch[i]
            bias_grad += self.bias_grad_batch[i]

        self.weights -= learning_rate * self.adam_w.update(weights_grad /BATCH_SIZE)
        self.bias -= learning_rate * self.adam_b.update(bias_grad /BATCH_SIZE)
        self.bias_grad_batch.clear()
        self.weights_grad_batch.clear()


class FCLayer():
    '''
    basic class for each layer
    '''
    def __init__(self, input_length, 
                 output_length,  
                 activator,
                 learning_rate):
        self.input_width = input_length
        self.output_length = output_length
        self.filter_width = input_length
        self.filter_height = output_length
        self.output_width = output_length
        self.output_array = np.zeros((self.output_length, 1)) # output_array == Z
        self.output_array_act = np.zeros((self.output_length, 1))
        self.dZ=np.zeros((self.output_length, 1))
        self.filter = Filter(self.filter_height,self.filter_width)
        self.delta_array = self.create_delta_array()
        self.activator = activator
        self.learning_rate = learning_rate
        self.parameters = [self.filter.weights, self.filter.bias]
        self.time = 0

    def forward(self, input_array):
        '''
        calculate the output
        and save the result on the self.output_array
        '''
        self.input_array = input_array
        self.output_array  = np.dot (self.filter.weights, self.input_array) + self.filter.bias
        self.output_array_act = self.output_array
        
        self.output_array_act = self.activator.forward(self.output_array)
    
    
    

    def backward(self, input_array, sensitivity_array):
        '''
        receive the delta array of this layer and save as sensitive array,
        calculate the delta array of the former layer 
        '''
        self.dz_calculate(sensitivity_array)                        
        self.linear_backward()
        self.filter.weights_grad_batch.append(self.filter.weights_grad)
        self.filter.bias_grad_batch.append(self.filter.bias_grad)

    def dz_calculate(self, sensitivity_array):
        '''
        calculate dZ
        '''
        dA= sensitivity_array
        for i in range(len(dA)):
            dA[i][0]  =dA[i][0] * self.activator.backward(self.output_array[i][0])
        self.dZ = dA


    def linear_backward(self):
        '''
        Do the backpropagation and calculate the gradients
        Z=WA+b
	    dZ: Upstream derivative, the shape (n^[l+1],m)
	    A: input of this layer 
	    
	    '''
        dZ= self.dZ
        A, W, b = self.input_array, self.filter.weights, self.filter.bias
        self.filter.weights_grad = np.dot(dZ, A.T)
        
        self.filter.bias_grad = dZ
        self.delta_array = np.dot(W.T, dZ)

    def update(self):
        '''
        update the filter
        '''
        self.filter.step(self.learning_rate)


    def create_delta_array(self):
        return np.zeros((self.filter_width,self.filter_height))
    

    