
import numpy as np

class ReluActivator(object):
    def forward(self, weighted_input):
        
        weighted_input = np.array(weighted_input)
        weighted_input[weighted_input <0 ]= 0
        
        return weighted_input

    def backward(self, output):
        return 1 if output > 0 else 0


class IdentityActivator(object):
    def forward(self, weighted_input):
        return weighted_input

    def backward(self, output):
        return 1

