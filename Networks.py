import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle

class Network:
    def __init__(self):
        self.layers=[]
        self.loss=None
        self.loss_prime=None
    
    def add(self,layer):
        self.layers.append(layer)
        print('Successful')
        
    
    def setup_loss(self,loss,loss_prime):
        self.loss=loss
        self.loss_prime=loss_prime
    
    def predict(self,input):
        n=len(input)
        result_his=[]
        for i in tqdm(range(n)):
            output=input[i]
            for layer in self.layers:
                output=layer.forward_propagation(output)
            result_his.append(output)
        
        return result_his
    
    
    def fit(self,X_train,Y_train,learning_rate,epochs):
        n=len(X_train)
        for i in range(1,epochs+1):
            err=0.0
            for j in tqdm(range(n)):
                output=X_train[j]
                # forward
                for layer in self.layers:
                    output=layer.forward_propagation(output)

                #estimate error of each sample
                
                err += self.loss(Y_train[j],output)
                #calculate error to backward
                error=self.loss_prime(Y_train[j],output)*2
                
                for layer in reversed(self.layers):
                    error= layer.backward_propagation(error,learning_rate)
            
            err=err/n
            print('epochs : {}/{} err={}'.format(i,epochs,err))  

    def fit_train_mini_batch(self,X_train,Y_train,learning_rate,epochs,size_batch):
        n=len(X_train)
        time=int(n/size_batch)
        for i in range(1,epochs+1):
            print('epochs : {}/{}'.format(i,epochs))
            
            X_train,Y_train=shuffle(X_train,Y_train)
            
            position=0
            # run the number of batch
            for j in range(1,time+1):
                err=0.0
                for z in tqdm(range(size_batch)):
                    output=X_train[position]
                    
                    for layer in self.layers:
                        output=layer.forward_propagation(output)
                    
                    err += self.loss(Y_train[position],output)
                    
                    error=self.loss_prime(Y_train[position],output)*2
                    
                    for layer in reversed(self.layers):
                        error=layer.backward_propagation(error,learning_rate)
                    
                    position += 1
                        
                err=err/size_batch
                print('Mini batch : {}/{} err={}'.format(j,time,err))  
        
        
        