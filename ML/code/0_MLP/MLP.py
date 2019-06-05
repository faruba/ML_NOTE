#!/usr/bin/env python
# coding: utf-8

# 初始化

import time
import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
#from testCases_v2 import *
#from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward
#from dnn_app_utils_v2 import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


np.random.seed(1)

def sigmoid(Z):
    temp =  1. /( 1. + np.exp(-Z))
    #print("sigmoid", Z, temp)
    return temp

active_func ={
    "relu": lambda Z :np.maximum(0, Z),
    "sigmoid": lambda Z :1. / (1. + np.exp(-Z)),
    "tanh":lambda Z :np.tanh(Z),
}

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

#虽然传A 比传Z 好,但是relu 需要Z才能求导,所以为了接口的统一性
#统一传Z
def sigmoid_backward(dA, Z):
    A = 1./(1.+np.exp(-Z))
    dZ = dA * A * (1.-A)
    assert (dZ.shape == Z.shape)
    return dZ

def tanh_backward(dA, Z):
    A = np.tanh(Z)
    s = 1 - (A ** 2)
    dZ = dA * s
    assert (dZ.shape == Z.shape)
    return dZ


dactive_func ={
    "relu": relu_backward,
    "sigmoid": sigmoid_backward,
    "tanh": tanh_backward,    
}

class MLP:
    def __init__(self, dim_conf):
        self.layer_dim= dim_conf
        self.init_params_2(dim_conf)

    def init_params_2(self, layer_dim):
        params = {}
        for l in range(len(layer_dim)-1):
            params["W"+str(l+1)] = np.random.normal(
                0,
                1/(layer_dim[l][0] ** 0.5),
                (layer_dim[l+1][0],layer_dim[l][0])
            )
            params["b"+str(l+1)] = np.zeros((layer_dim[l+1][0],1))
        self.params = params
        print("====init 2", self.params)
            
    def init_params(self, layer_dim):
        params = {}
        for l in range(len(layer_dim)-1):
            params["W"+str(l+1)] = np.random.randn(
                layer_dim[l+1][0],layer_dim[l][0]
            )*0.01
            params["b"+str(l+1)] = np.zeros((layer_dim[l+1][0],1))
        #print("====init 1",params)
        self.params = params
 
    def forward(self,X):
        caches ={}
        current_input = X
        caches["A0"] = X
        for l in range(len(self.layer_dim)-1):
            str_l = str(l+1)
            caches["Z"+str_l] = np.dot(self.params["W"+str_l],current_input)
            + self.params["b"+str_l]
            
            current_input = caches["A"+str_l] = active_func[self.layer_dim[l+1][1]](caches["Z"+str_l])
            #print("=== for", str_l, current_input, caches["Z"+str_l])
            
            #print("======== forward", str_l, caches["Z"+str_l].shape, caches["A"+str_l].shape, self.params["W"+str_l].shape)
        return current_input,caches
        
    def cost(self, AL, Y):
        m = AL.shape[1]
        cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())
        
        return cost

    #z^[l] = W_l a^[l-1]  + b_l a^[l] = fa(z^[l])
    def linear_backward(self, dZ, cache,l):
        strL = str(l)
        prev_A = cache["A"+str(l-1)]
        W =  self.params["W"+strL]
        b =  self.params["b"+strL]

        m = prev_A.shape[1]

        dw = np.dot(dZ,prev_A.T)/m
        db = np.sum(dZ,axis=1,keepdims=True)/m
        dA_pre = np.dot(W.T,dZ)
        assert (dA_pre.shape == prev_A.shape)
        assert (dw.shape == W.shape)
        assert (db.shape == b.shape)
        return  dA_pre, dw, db
 
    def backward(self, AL, Y):
        grads={}
        Y = Y.reshape(AL.shape)
        #print("====b", Y, AL)
        dAL = - (np.divide(Y,AL)-np.divide(1-Y,1-AL))
        #print("====DA", dAL)
        current_d = dAL
        for l in reversed(range(1, len(self.layer_dim))):
            strL=str(l)
            dZ = dactive_func[self.layer_dim[l][1]](current_d,self.caches["Z"+strL])
            current_d, dw, db = self.linear_backward(dZ,self.caches,l)
            #grads["dZ"+strL] = dz
            grads["dW"+strL] = dw
            grads["db"+strL] = db
            #grads["dA"+strL] =  = dA
        return grads      

    def update_param(self, grads, learn_rate):
        for l in range(len(self.layer_dim)-1):
            strL = str(l+1)
            #print(strL, self.params["W"+strL].shape,grads["dW"+strL].shape)
            self.params["W"+strL] -= (learn_rate*grads["dW"+strL])
            self.params["b"+strL] -= (learn_rate*grads["db"+strL])

    def train(self, X, Y, times, learn_rate, debug = None):
        for t in range(0, times):
            AL, caches = self.forward(X)
            #print("===> ", caches, AL)
            self.caches = caches
            if debug != None and t % 100 == 0:
                debug["cost"](self.cost(AL,Y))
            grads = self.backward(AL,Y)
            #print("<=== ", grads)
            self.update_param(grads, learn_rate)
        
        #return self.params
    def predict(train_x, train_y):
        p = np.zeros((1,train_x.shape[1]))
        AL, caches = self.forward(train_x)
        for i in range(0, AL.shape[1]):
            p[0][i] = AL[0,i] > 0.5 ? 1 : 0

        return 
        
#
#
## In[53]:
#
#
#train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
#
#train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   
## The "-1" makes reshape flatten the remaining dimensions
#test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
#
## Standardize data to have feature values between 0 and 1.
#train_x = train_x_flatten/255.
#test_x = test_x_flatten/255.
#
#print ("train_x's shape: " + str(train_x.shape))
#print ("test_x's shape: " + str(test_x.shape))
#
#
## In[ ]:
#
#
#class Debug():
#
#
## In[ ]:
#
#
#two_layer_model = train(train_x, train_y, 
#                        layers_dims = (n_x, n_h, n_y), 
#                        num_iterations = 3000, print_cost=True)
#
