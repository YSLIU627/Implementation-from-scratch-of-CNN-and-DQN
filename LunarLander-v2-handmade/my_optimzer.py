import numpy as np

class Adam():
    def __init__(self, shape):
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.eps = 1e-8
        self.shape = shape
        self.t = 0
        self.t_ =0
        self.v = np.zeros(self.shape)
        self.m = np.zeros(self.shape)
        self.g = np.zeros(self.shape)
        self.theta = np.zeros(self.shape)
    
    def update (self, gradient):
        self.t += 1
        self.g= gradient
        self.m = self.beta_1* self.m +(1-self.beta_1) * self.g
        self.v = self.beta_2 * self.v +(1-self.beta_2) * ((self.g )**2)
        m_bar = self.m / (1-(self.beta_1)** self.t) 
        v_bar = self.v/ (1-self.beta_2 ** self.t)
        self.theta = m_bar /(v_bar ** 0.5 +self.eps) 
        return self.theta