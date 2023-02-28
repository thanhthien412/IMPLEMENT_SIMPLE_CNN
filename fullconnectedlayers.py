from layers  import Layer
import numpy as np

class FClayers(Layer):
    def __init__(self,input_shape,output_shape):
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.weights=np.random.rand(input_shape[1],output_shape[1])-0.5
        self.bias=np.random.rand(1,output_shape[1])-0.5
    
    def forward_propagation(self, input,W=None):
        self.input=input
        self.output=np.dot(self.input,self.weights)+ self.bias
        return self.output
    
    def backward_propagation(self, output_error, learning_rate,W=None):
        current_layers_err=output_error.dot(self.weights.T) # we divide formula into 2 part FClayers and active function this will be used in the next layer
        gra_w=np.dot(self.input.T,output_error)
        self.weights -= gra_w*learning_rate
        self.bias    -= output_error*learning_rate
        
        return current_layers_err
        