from layers import Layer
import numpy as np
from check_function import check


class MaxPooling(Layer):
    def __init__(self,Pooling_shape,stride=1):
        self.Pooling_shape=Pooling_shape
        self.stride=stride
        self.R_size=None
        self.C_size=None
    
    def forward_propagation(self, input,W=None):
        self.input=input
        self.R_size,self.C_size=check(input.shape,self.Pooling_shape,self.stride)
        result=np.ones((self.R_size,self.C_size,input.shape[2]),dtype=float)
            
        for j in range(input.shape[2]):
            r=0
            R_time=0
            while R_time < self.R_size: # we can change to while when stride different from 1
                c=0
                C_time=0 
                while C_time< self.C_size:
                    result[R_time,C_time,j]=np.max(input[r:r+self.Pooling_shape[0],c:c+self.Pooling_shape[1],j])
                    C_time += 1
                    c += self.stride
                r +=  self.stride
                R_time +=1
        
        self.output=result
        result[result<0]=0
        return result

    def backward_propagation(self, output_error, learning_rate,W=None):
        for j in range(output_error.shape[2]):
            r=0
            R_time=0
            while R_time < self.R_size:
                c=0
                C_time=0
                while C_time < self.C_size:
                    self.input[r:r+self.Pooling_shape[0],c:c+self.Pooling_shape[1],j][self.input[r:r+self.Pooling_shape[0],c:c+self.Pooling_shape[1],j]==self.output[R_time,C_time,j]]=output_error[R_time,C_time,j]
                    C_time +=1
                    c += self.stride
                r+= self.stride
                R_time += 1
                                
        return self.input