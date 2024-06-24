#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import random
import pprint
import copy


# In[9]:


data = open('dinos.txt', 'r').read()
data = data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('Total Characters', data_size)
print('Unique Characters', vocab_size)
n_x = vocab_size


# In[10]:


chars = sorted(chars)
chars


# In[11]:


char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for i,ch in enumerate(chars)}
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(ix_to_char)


# In[12]:


n_a = 100 
seq_length = 25 
learning_rate = 1e-1


# In[13]:


Wax = np.random.randn(n_a, n_x)*0.01
Waa = np.random.randn(n_a, n_a)*0.01
Wya = np.random.randn(n_x, n_a)*0.01
b = np.zeros((n_a, 1))
by = np.zeros((n_x, 1))


# In[14]:


def lossFun(inputs, targets, a_prev):
    
  x, a, y, p = {}, {}, {}, {}
  a[-1] = np.copy(a_prev)
  loss = 0
  # forward pass
  for t in range(len(inputs)):
    x[t] = np.zeros((n_x,1))
    x[t][inputs[t]] = 1
    a[t] = np.tanh(np.dot(Wax, x[t]) + np.dot(Waa, a[t-1]) + b) 
    y[t] = np.dot(Wya, a[t]) + by 
    p[t] = np.exp(y[t]) / np.sum(np.exp(y[t])) 
    loss += -np.log(p[t][targets[t],0])
  # backward pass:
  dWax, dWaa, dWya = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
  db, dby = np.zeros_like(b), np.zeros_like(by)
  danext = np.zeros_like(a[0])
  for t in reversed(range(len(inputs))):
    dy = np.copy(p[t])
    dy[targets[t]] -= 1 
    dWya += np.dot(dy, a[t].T)
    dby += dy
    da = np.dot(Wya.T, dy) + danext 
    daraw = (1 - a[t] * a[t]) * da 
    db += daraw
    dWax += np.dot(daraw, x[t].T)
    dWaa += np.dot(daraw, a[t-1].T)
    danext = np.dot(Waa.T, daraw)
  for dparam in [dWax, dWaa, dWya, db, dby]:
    np.clip(dparam, -5, 5, out=dparam) 
  return loss, dWax, dWaa, dWya, db, dby, a[len(inputs)-1]


# In[15]:


def sample(a, seed_ix, n):
    
  x = np.zeros((n_x, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a) + b)
    y = np.dot(Wya, a) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(n_x), p=p.ravel())
    x = np.zeros((n_x, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes


# In[17]:


n, p = 0, 0
mWax, mWaa, mWya = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
mb, mby = np.zeros_like(b), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/n_x)*seq_length # loss at iteration 0
while n <= 10000:
  if p+seq_length+1 >= len(data) or n == 0: 
    a_prev = np.zeros((n_a,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 100 == 0:
    sample_ix = sample(a_prev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print ('----\n %s \n----' % (txt, ))

  # forward seq_length characters through the net and fetch gradient
  loss, dWax, dWaa, dWya, db, dby, a_prev = lossFun(inputs, targets, a_prev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print ('iter %d, loss: %f' % (n, smooth_loss)) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wax, Waa, Wya, b, by], 
                                [dWax, dWaa, dWya, db, dby], 
                                [mWax, mWaa, mWya, mb, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[126]:


#Use input data to get these
n_a = 10
n_x = 10
n_y = 10
m = 20
Tx = 8


# In[127]:


#Use input data to get these
xt = np.random.randn(n_x,m)
a_prev= np.random.randn(n_a,m)


# In[128]:


#Use input data to get these
Waa = np.random.randn(n_a,n_a)
Wax = np.random.rand(n_a,n_x)
Wya = np.random.randn(n_y,n_a)
ba = np.random.randn(n_a,1)
by = np.random.randn(n_y,1)
x = np.random.randn(n_x,m,Tx)


# In[129]:


parameters = {
    "Waa": Waa,
    "Wax": Wax,
    "Wya": Wya,
    "ba": ba,
    "by": by
}


# In[130]:


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# In[131]:


def rnn_cell_forward(xt, a_prev, parameters):
    
    Waa = parameters["Waa"]
    Wax = parameters["Wax"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    a_next = np.tanh(np.dot(Waa,a_prev)+np.dot(Wax,xt)+ba)
    yt_pred = softmax(np.dot(Wya,a_next)+by)
    
    cache = (a_next, a_prev, xt, parameters)
    return a_next, yt_pred, cache


# In[132]:


def rnn_forward(x,a0,parameters):
    #Get dims
    n_x,m,Tx = x.shape
    n_y,n_a = parameters["Wya"].shape
    
    #initialize
    caches = []
    a = np.zeros((n_a,m,Tx))
    y_pred = np.zeros((n_y,m,Tx))
    a_next = a0
    #loop
    for t in range(Tx):
        a[:,:,t] , y_pred[:,:,t], cache = rnn_cell_forward(x[:,:,t], a_next, parameters)
        caches.append(cache)
    caches = (caches, x)
    return a, y_pred, caches


# In[133]:


def rnn_cell_backward(da_next, cache):
    
    (a_next, a_prev, xt, parameters) = cache
    
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    dtanh = np.multiply(da_next, 1 -pow(np.tanh(np.dot(Waa,a_prev)+np.dot(Wax,xt)+ba),2))

    dxt = np.dot(Wax.T,dtanh)
    dWax = np.dot(dtanh,xt.T)

    da_prev = np.dot(Waa.T,dtanh)
    dWaa = np.dot(dtanh,a_prev.T)

    dba = np.sum(dtanh,axis=1, keepdims=True)

    
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}
    
    return gradients


# In[134]:


def rnn_backward(da, caches):

    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0]
    
    n_a, m, T_x = da.shape
    n_x, m = x1.shape 
    
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    
    for t in reversed(range(T_x)):
        gradients = rnn_cell_backward(da[:,:,t] + da_prevt, caches[t])
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
        dx[:, :, t] = dxt  
        dWax += dWaxt  
        dWaa += dWaat  
        dba += dbat  
        
    da0 = da_prevt

    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa,"dba": dba}
    
    return gradients


# In[135]:


data = open('dinos.txt', 'r').read()
data = data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('Total Characters', data_size)
print('Unique Characters', vocab_size)


# In[136]:


chars = sorted(chars)
chars


# In[137]:


char_to_ix = {ch:i for i,ch in enumerate(chars)}
ix_to_char = {i:ch for i,ch in enumerate(chars)}
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(ix_to_char)


# GRADIENT CLIPPING

# In[138]:


def clip(gradients, maxValue):
    gradients = copy.deepcopy(gradients)
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
    for gradient in gradients:
        np.clip(gradients[gradient], -maxValue, maxValue, out = gradients[gradient])
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    return gradients


# SAMPLING

# In[139]:


def sample(parameters, char_to_idx, seed):
    
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    
    x = np.zeros((vocab_size,1))
    a_prev = np.zeros((n_a,1))
    
    indices = []
    
    idx = -1
    
    counter = 0
    newline_character = char_to_ix['\n']
    
    while (idx != newline_character and counter != 50):
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)
        
        np.random.seed(counter + seed)
        idx = np.random.choice(range(vocab_size), p=y.ravel())
        
        indices.append(idx)
        
        x=np.zeros((vocab_size,1))
        x[idx] = 1
        
        a_prev = a
        
        seed+=1
        counter+=1
    
    if(counter==50):
        indices.append(char_to_index['\n'])
        
    return indices


# In[140]:


def RNN_forward(X, Y, a_prev, parameters):
    n_x,m,Tx = X.shape
    n_y,n_a = parameters["Wya"].shape
    
    caches = []
    a = np.zeros((n_a,m,Tx))
    y_pred = np.zeros((n_y,m,Tx))
    a_next = a_prev
    loss = 0
    
    for t in range(Tx):
        a[:,:,t] , y_pred[:,:,t], cache = rnn_cell_forward(x[:,:,t], a_next, parameters)
        caches.append(cache)
        loss += -np.sum(Y[:,:,t] * np.log(yt_pred[:,:,t]))
    caches = (caches, X, Y)
    return loss, caches


# In[141]:


def RNN_backward(X, Y, parameters, caches):
    
    
    # Retrieve parameters and cached values
    Waa, Wax, Wya, b, by = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['b'], parameters['by']
    (caches, X, Y) = caches
    n_x, m, T_x = X.shape
    n_y,n_a = parameters["Wya"].shape    
    
    # Initialize gradients with zero
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dWya = np.zeros((n_y,n_a))
    dba = np.zeros((n_a, 1))
    da0 = np.zeros((n_a, m))
    dby = np.zeros((n_y,1))
    da_prevt = np.zeros((n_a, m))
        
    # Backpropagation through time
    for t in reversed(range(T_x)):
        gradients = rnn_cell_backward(da[:,:,t] + da_prevt, caches[t])
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
        dx[:, :, t] = dxt  
        dWax += dWaxt  
        dWaa += dWaat  
        dba += dbat  
        
    da0 = da_prevt

    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa,"dba": dba}
    
    return gradients, da0


# In[142]:


def update_parameters(parameters, gradients, learning_rate):
    
    updated_parameters = {}
    for param_name, param_value in parameters.items():
        gradient = gradients[param_name]
        updated_param_value = param_value - learning_rate * gradient
        updated_parameters[param_name] = updated_param_value
    return updated_parameters


# In[143]:


def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    
    loss, cache = RNN_forward(X, Y, a_prev, parameters)
    
    gradients, a = RNN_backward(X, Y, parameters, cache)
    
    gradients = clip(gradients, maxValue=5)
    
    parameters = update_parameters(parameters, gradients, learning_rate)
        
    return loss, gradients, a[len(X)-1]


# In[ ]:


def model():
    
    
    
    return parameters, last_dino_name


# In[ ]:




