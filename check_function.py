import numpy as np

def check(input_shape,shape,stride):
    c=r=0
    c_time=0
    r_time=0
    while (r<input_shape[0]):
        if(r+shape[0]<input_shape[0]):
            r += stride
            r_time +=1
        else:
            break   
    while(c<input_shape[1]):
        if(c+shape[1]<input_shape[1]):
            c += stride
            c_time +=1
        else:
            break
    return r_time+1,c_time+1
