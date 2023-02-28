
from matplotlib.cbook import print_cycles
from layers import Layer
import numpy as np
from check_function import check



class Con2D(Layer):
    def __init__(self,Con2D_shape,filter,stride=1):
        self.Con2D_shape=Con2D_shape
        self.filter=filter
        self.stride=stride
        self.bias=np.random.randint(0,3,size=(self.filter))
        self.Con2D_matrix=np.random.randint(-2,2,(self.Con2D_shape[0],self.Con2D_shape[1],filter))
        self.Con2D_matrix=self.Con2D_matrix.astype(np.float64)
        self.bias=self.bias.astype(np.float64)
    
    def relu_prime(self,z):
        z[z<=0]=0
        z[z>0]=1
        return z
    
    def forward_propagation(self, input,W=None):
        self.input=input
        self.R_size,self.C_size=check(input.shape,self.Con2D_shape,self.stride)
        result=np.ones((self.R_size,self.C_size,self.filter),dtype=np.float64)
        bias=0
        input_depth=input.shape[2]
        for i in range(self.filter):
            matrix_temp=self.Con2D_matrix[:,:,i]
            temp=np.ones((self.R_size,self.C_size),dtype=np.float64)
            r=0
            R_time=0
            while R_time <self.R_size:
                c=0
                C_time=0
                while C_time < self.C_size:
                    sum=0.0
                    for j in range(input_depth):
                        sum += np.sum(input[r:r+self.Con2D_shape[0],c:c+self.Con2D_shape[1],j]*matrix_temp,dtype=np.float64)
                    temp[R_time,C_time]=(sum+self.bias[bias])
                    C_time += 1
                    c += self.stride
                R_time +=1
                r += self.stride
            result[:,:,i]=temp
            bias += 1
        self.output=result 
        result[result<0]=0
        return result
    
    
    def backward_propagation(self, output_error, learning_rate,W=None):
        output_error= self.relu_prime(self.output)*output_error     
        loss_matrix=self.output-output_error
        input_depth=self.input.shape[2]
        # create predicted input
        number_elements_in_matrix=self.Con2D_shape[0]*self.Con2D_shape[1]
        magnitude_matrix=np.array(self.Con2D_matrix)
        magnitude_matrix[magnitude_matrix<0]= -magnitude_matrix[magnitude_matrix<0]
        new_input=np.array(self.input,dtype=np.float64)
        
        for i in range(self.filter):
            matrix_temp=self.Con2D_matrix[:,:,i]
            sum=np.sum(magnitude_matrix[:,:,i])
            r=0
            R_time=0
            while R_time < self.R_size:
                c=0
                C_time=0
                while C_time < self.C_size:
                    for j in range(input_depth):
                        r_temp=0
                        for x in range(r,r+self.Con2D_shape[0]):
                            c_temp=0
                            for y in range(c,c+self.Con2D_shape[1]):  
                                new_input[x,y,j] = new_input[x,y,j] - ((loss_matrix[R_time,C_time,i])/(sum*number_elements_in_matrix*self.filter*input_depth))*matrix_temp[r_temp,c_temp]
                                c_temp +=1
                            r_temp +=1
                    C_time += 1
                    c += self.stride
                R_time += 1
                r += self.stride
                            
        # fit weight
        number_elements=output_error.shape[0]* output_error.shape[1]
        
        for i in range(self.filter):
            r=0
            R_time=0
            bias=0
            gra=np.zeros((self.Con2D_shape[0],self.Con2D_shape[1]),dtype=np.float64)
            while R_time < self.R_size:
                c=0
                C_time=0
                
                while C_time < self.C_size:
                    temp_gra=np.zeros((self.Con2D_shape[0],self.Con2D_shape[1]),dtype=np.float64)
                    
                    for j in range(input_depth):
                        temp_gra += 2* loss_matrix[R_time,C_time,i] * self.input[r:r+self.Con2D_shape[0],c:c+self.Con2D_shape[1],j]
                    
                    gra += temp_gra/ input_depth
                    C_time += 1
                    c += self.stride
                    
                R_time +=1
                r += self.stride
            self.Con2D_matrix[:,:,i] -= (gra/number_elements) * learning_rate
            self.bias[bias] -= (np.sum(loss_matrix[:,:,i])*learning_rate)/number_elements
            bias += 1
        
        return new_input            