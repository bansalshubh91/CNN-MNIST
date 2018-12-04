
# coding: utf-8

# In[ ]:


import numpy as np
import h5py
import time
import copy
from random import randint
from scipy import signal

#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()


#Implementation of stochastic gradient descent algorithm
#number of inputs
num_inputs = 28
#each filter_dimension
filter_dims = 5

#number of channels/filters
num_channels = 3

#number of outputs
num_outputs = 10
model = {}

model['K'] = np.random.randn(filter_dims, filter_dims, num_channels) / num_inputs

model['W'] = np.random.randn(num_outputs, num_inputs-filter_dims+1, num_inputs-filter_dims+1, num_channels) / num_inputs
model['b'] = np.random.randn(num_outputs,1) / num_outputs

model_grads = copy.deepcopy(model)

def conv(X,K,i,j):
    a=0
    for m in range(len(K)):
        for n in range(len(K)):
            a = a + K[m,n]*X[i+m,j+n]    
    return a


def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ
def forward(x,y,model):
    Z = np.random.randn(num_inputs-filter_dims+1, num_inputs-filter_dims+1, num_channels) 
    for c in range(num_channels):
        for i in range(num_inputs - filter_dims + 1):
            for j in range(num_inputs - filter_dims + 1):
                Z[i,j,c] = conv(x,model['K'][:,:,c],i,j)
      
    
    activ_deriv = (Z >= 0).astype(int)
    

                
    H = np.maximum(Z,0)            
            

    
    U = np.tensordot(model['W'],H, 3).reshape((num_outputs,1))
    U = U + model['b']
    p = softmax_function(U)
    return p,H,activ_deriv 
def backward(x,y,p,H,activ_deriv, model, model_grads):
    dU = 1.0*p
    dU[y] = dU[y] - 1.0
    
    model_grads['W'] = np.tensordot(dU,H.reshape((1,num_inputs-filter_dims+1, num_inputs-filter_dims+1, num_channels)),1)

    
    delta = np.tensordot(np.transpose(dU),model['W'],1).reshape(H.shape)
    model_grads['b'] = dU    
    for c in range(num_channels):
        for i in range(filter_dims):
            for j in range(filter_dims):
                model_grads['K'][i,j,c] = conv(x,delta[:,:,c]*activ_deriv[:,:,c],i,j)

    return model_grads


import time
time1 = time.time()
LR = 0.01
num_epochs = 15
for epochs in range(num_epochs):
    #Learning rate schedule
    if (epochs > 5):
        LR = 0.001
    if (epochs > 10):
        LR = 0.0001
    if (epochs > 15):
        LR = 0.00001
    total_correct = 0
    for n in range( len(x_train)):
        n_random = randint(0,len(x_train)-1 )
        y = y_train[n_random]
        x = x_train[n_random][:].reshape((28,28))
        p,H,activ_deriv = forward(x, y, model)
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1
        model_grads = backward(x,y,p,H,activ_deriv, model, model_grads)
        model['W'] = model['W'] - LR*model_grads['W']
        model['K'] = model['K'] - LR*model_grads['K']
        model['b'] = model['b'] - LR*model_grads['b']

    print(total_correct/np.float(len(x_train) ) )
time2 = time.time()
print(time2-time1)
######################################################
#test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = x_test[n][:].reshape((28,28))
    p,H,activ_deriv = forward(x, y, model)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
print(total_correct/np.float(len(x_test) ) )



# In[143]:





# In[144]:





# In[145]:





# In[147]:





# In[148]:





# In[50]:





# In[51]:




