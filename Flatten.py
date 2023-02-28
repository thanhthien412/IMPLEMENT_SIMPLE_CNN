
from layers import Layer
import numpy as np



class Flatten(Layer):
    def __init__(self):
        pass
    
    def forward_propagation(self, input):
        self.input_shape=input.shape
        shape=input.shape
        result=[]
        for z in range(shape[2]):
            for x in range(shape[0]):
                for y in range(shape[1]):
                    result.append(input[x,y,z])
        
        return np.array([result])
    
    def backward_propagation(self, output_error, learning_rate):
        result=np.ones(self.input_shape,dtype=np.float64)
        position=0
        for z in range(result.shape[2]):
            for x in range(result.shape[0]):
                for y in range(result.shape[1]):
                    result[x,y,z]=output_error[0][position]
                    position += 1             
        
        return result