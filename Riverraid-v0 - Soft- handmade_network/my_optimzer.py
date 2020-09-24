import numpy as np
import torch
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
class Adam():
    '''
    Handmade Adam Optimizer
    '''
    def __init__(self, shape ,learning_rate):
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.eps = 1e-8
        self.shape = shape
        self.t = 0
        self.t_ =0
        self.v = torch.zeros(self.shape,device=device)
        self.m = torch.zeros(self.shape,device=device)
        self.g = torch.zeros(self.shape,device=device)
        self.theta = torch.zeros(self.shape,device=device)
        self.learning_rate = learning_rate
    
    def update (self, gradient):
        self.t += 1
        self.g= gradient.to(device)
        self.m = self.beta_1* self.m +(1-self.beta_1) * self.g
        self.v = self.beta_2 * self.v +(1-self.beta_2) * ((self.g )**2)
        m_bar = self.m / (1-(self.beta_1)** self.t) 
        v_bar = self.v/ (1-self.beta_2 ** self.t)
        self.theta = m_bar /(v_bar ** 0.5 +self.eps) 
        
        return self.theta * self.learning_rate