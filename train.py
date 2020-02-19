#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from numpy import linalg as LA
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF
from utils.preprocessing import generate_train_data
from utils.optim import *
from utils.metric import get_mse
import sys
import os


"""
In this sample case, we get rid of data preprocessing step and we only give optimization steps, for movielens dataset, user owns two level hierarchical structure, and item owns two level hierarchical structure, so we have n:number of users; m: number of items; n1:group number of users; m1: group number of items.

After data preprocessing, we have already have X(user flat feature), Y(item flat feature), P_1(user hierarchy), Q_1(item hierarchy)

z: dimension of user flat feature

d: hidden vector dimension for matrix factorization

q: dimension of item flat feature
"""

def initialize(z,d,q):
    S1 = np.random.rand(z,d)
    S2 = np.random.rand(q,d)
    W1 = np.random.rand(z,z)
    W2 = np.random.rand(q,q)
    return S1,S2,W1,W2

class HIRE():
    def __init__(self, user_flat_feature, item_flat_feature, user_hierarchy, item_hierarchy,
    																					gamma=0.5, theta=0.5, corrupted_rate=0.2, beta=0.5, lamda=1,alpha=0.5,d_hidden=50):
        # user flat features
        self.u_flat = user_flat_feature
        self.i_flat = item_flat_feature
        self.u_hier = user_hierarchy
        self.i_hier = item_hierarchy
        self.gamma, self.theta, self.corrupted, self.beta, self.lamda, self.alpha, self.d = gamma,theta, corrupted_rate, beta, lamda, alpha, d_hidden
        self.z = user_flat_feature.shape[0]
        self.q = item_flat_feature.shape[0]
        self.n1 = user_hierarchy.shape[0]
        self.n = user_hierarchy.shape[1]
        self.m1 = item_hierarchy.shape[1]
        self.m = item_hierarchy.shape[0]

    def train(self, sigma, train_data, test_data,optim_steps=20000, verbose=True):
    	# initialize
    	S1, S2, W1, W2 = initialize(self.z, self.d, self.q)

    	train_loss_record = []
    	test_loss_record = []
    	
    	model_nmf = NMF(n_components=self.d, init='random')
    	U = model_nmf.fit_transform(train_data)
    	V = model_nmf.components_

    	model_2_nmf = NMF(n_components=self.m1, init = 'random')
    	V2 = model_2_nmf.fit_transform(V)
    	V1 = model_2_nmf.components_

    	model_3_nmf = NMF(n_components=self.n1, init = 'random')
    	U1 = model_3_nmf.fit_transform(U)
    	U2 = model_3_nmf.components_

    	# if loss increase for five future steps, stop
    	stop_time = 0
    	times, old_loss = 0, 100000
    	for _ in range(optim_steps):
    		U = U1.dot(U2)
    		V = V2.dot(V1)
    		pred = U.dot(V)
    		loss = get_mse(pred,train_data)
    		loss_test = get_mse(pred, test_data)

    		if loss_test > old_loss:
    			if stop_time == 5:
    				break
    			else:
    				stop_time +=1
    		else:
    			stop_time = 0

    		old_loss = loss_test
    		if verbose:
    			if times%1000 == 0:
    				print("[Info] At time-step {}, test data mse loss is {}".format(times,loss_test))

    		W1 -= Lipschitz_W1(self.u_flat, self.corrupted, self.gamma, self.z) * SGD_W1(self.u_flat, self.corrupted, self.gamma, S1, U, self.z, W1)
    		W2 -= Lipschitz_W2(self.i_flat, self.corrupted, self.theta, self.q) * SGD_W2(V, self.corrupted, self.theta, S2, self.i_flat, W2, self.q)
    		S1 -= Lipschitz_S1(U) * SGD_S1(W1, self.u_flat, U, self.gamma, S1)
    		S2 -= Lipschitz_S2(V) * SGD_S2(W2, self.i_flat, V, S2, self.theta)
    		V1 -= Lipschitzz_V1(U1, U2, V2, self.lamda, self.beta, self.i_hier, S2, self.theta) * SGD_V1(U1, U2, V2, sigma, V1, train_data, self.lamda, self.beta, self.i_hier, self.theta, S2, self.i_flat,W2)
    		V2 -= Lipschitzz_V2(U1,U2, self.beta, V1, self.i_hier, self.lamda, self.theta, S2, self.m1) * SGD_V2(U2, U1, sigma, V2, V1,train_data, self.beta, self.i_hier, self.m1, self.lamda, self.theta, S2, W2, self.i_flat)
    		U1 -= Lipschitzz_U1(U2, V2, V1, self.alpha, self.lamda, self.u_hier, S1, self.gamma) * SGD_U1(sigma, U1, U2, V1, V2,train_data,self.alpha, self.u_hier, self.lamda, self.gamma, S1, self.u_flat, W1)
    		U2 -= Lipschitzz_U2(U1, V1, V2, self.lamda, self.gamma, S1, self.u_hier, self.alpha, self.n1) * SGD_U2(U1, sigma, U2, V2, V1, train_data, self.beta, self.gamma, S1, self.u_flat, W1, self.alpha, self.u_hier, self.n1,self.lamda)
    		times += 1
    		train_loss_record.append(loss)
    		test_loss_record.append(loss_test)
    	return U, V, train_loss_record, test_loss_record

# a test demo
if __name__ == "__main__":
	base_path = os.path.dirname(os.path.realpath(__file__))

	user_flat_feature = np.loadtxt(os.path.join(base_path,"data","X.txt"))
	item_flat_feature = np.loadtxt(os.path.join(base_path,"data","Y.txt"))
	user_hierarchy = np.loadtxt(os.path.join(base_path,"data","user_hierarchy.txt"))
	item_hierarchy = np.loadtxt(os.path.join(base_path,"data","item_hierarchy.txt"))
	# we do five fold cross-validation here
	test_loss = []
	for fold in range(1,6):
		# you can grid search gamma, corrupted_rate, beta, lamda,alpha,d_hidden over here, for simplicity,  I set all values equal to 0.5, but these are definitely not the best hyper-parameters.
		HIRE_model = HIRE(user_flat_feature, item_flat_feature, user_hierarchy, item_hierarchy)
		sigma, train_data, test_data = generate_train_data(os.path.join(base_path,"data","ml-100k","u{}.base".format(fold)),os.path.join(base_path,"data","ml-100k","u{}.test".format(fold)))
		U,V, train_record, test_record = HIRE_model.train(sigma, train_data, test_data)
		test_loss.append(min(test_record))
	print("[Result] RMSE for method HIRE is {}".format(np.sqrt(np.mean(test_loss))))



















