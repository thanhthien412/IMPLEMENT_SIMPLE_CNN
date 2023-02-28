from layers import Layer

class ActivationLayer(Layer):
    def __init__(self,input_shape,output_shape,activation,activation_prime):
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.activation=activation
        self.activation_prime=activation_prime
    
    def forward_propagation(self, input,W=None):
        self.input=input
        self.output=self.activation(input)
        return self.output
    
    def backward_propagation(self, output_error, learning_rate,W=None):
        return self.activation_prime(self.input)*output_error
    
    