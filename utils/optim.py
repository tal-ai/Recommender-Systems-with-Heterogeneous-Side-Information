#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as LA

# as described in paper, we update parameters by stochastic gradient descent.
def SGD_W1(X,corrupted_rate,gamma,S_1,U,z,W1):
    '''
    Params
      Input:
        X: numpy array, dimension(z * n)
          Side information of users
        
        corrupted_rate: float
          corrupte probability of every side information, used to generate \bar{x} and \tlide{x}
          
        gamma: float
        
        S_1: np array, dimension(z * d)
          Projection matrix for U
        
        U: np array, dimension (n * d)
          Latent feature matrix of users
          
        z: dimension of user features
        
      Output:
        matrix, dimension(z * z)
    '''
    U = U.data
    term_1 = (1-corrupted_rate) * np.dot(X,np.transpose(X))
    term_1 += gamma * np.dot(S_1,np.dot(np.transpose(U),np.transpose(X)))
    term_2 = gamma * W1.dot(np.dot(X,np.transpose(X)))
    T_tmp = (1 - corrupted_rate) * (1 - corrupted_rate) * (np.ones([z, z]) - np.diag(np.ones([z]))) * np.dot(X, np.transpose(X))
    T_tmp += (1-corrupted_rate) * np.diag(np.ones([z])) * np.dot(X,np.transpose(X))
    term_2 += W1.dot(T_tmp)
    return term_2 - term_1
def SGD_S1(W_1,X,U,gamma,S_1):
    '''
    Params:
    
      Input:
        W_1: numpy array, dimension(z * z)
          Mapping function for X in auto-coder
        
        X: numpy array, dimension(z * n)
          Side information of users
          
        U: numpy array, dimension(n * d)
          latent features matrix of users
          
      output:
        matrix, dimension(z * d)
    '''
    U = U.data
    a = gamma*S_1.dot(np.transpose(U)).dot(U)
    b = gamma * W_1.dot(X).dot(U)
    return a-b

def SGD_W2(V,corrupted_rate,gamma,S_2,Y,W2,q):
    '''
    Params:
    
      Input:
        Y: numpy array, dimension(q * m)
          Side information of items
        
        corrupted_rate: float
          corrupte probability of every side information, used to generate \bar{x} and \tlide{x}
          
        gamma: float
        
        S_2: np array, dimension(q * d)
          Projection matrix for V
        
        V: np array, dimension (m * d)
          Latent feature matrix of items
          
        q: dimension of item features
    '''
    V = np.transpose(V)
    term_1 = (1-corrupted_rate)*np.dot(Y,np.transpose(Y))
    term_1 += gamma * (S_2.dot(np.transpose(V)).dot(np.transpose(Y)))
    term_2 = gamma * W2.dot(Y).dot(np.transpose(Y))
    T_tmp = (1-corrupted_rate) * (1-corrupted_rate) * (np.ones([q,q]) - np.diag(np.ones([q]))) * np.dot(Y, np.transpose(Y))
    T_tmp += (1-corrupted_rate) * np.diag(np.ones([q])) * np.dot(Y, np.transpose(Y))
    term_2 += W2.dot(T_tmp)
    return term_2-term_1
def SGD_S2(W_2,Y,V,S_2,gamma):
    '''
    Params:
      
      input:
        W_2: numpy array, dimension(q * q)
          Mapping function for Y in auto-coder
        
        Y: numpy array, dimension(q * m)
          Side information of items
          
        V: numpy array, dimension(m * d)
          latent features matrix of items
          
      Output:
        matrix, dimension(q * d)
    '''
    V = V.T
    return gamma*(S_2.dot(np.transpose(V))-W_2.dot(Y)).dot(V)
def SGD_V1(U1,U2,V2,sigma,V1,rating_matrix,lamda,beta,Q1,gamma,S2,Y,W2):
    term_1 = np.transpose(U1.dot(U2).dot(V2)).dot(sigma *(U1.dot(U2).dot(V2).dot(V1)-rating_matrix))
    term_2 = lamda* V1
    term_3 = beta*np.transpose(V2).dot(V2.dot(V1).dot(Q1)-V2).dot(np.transpose(Q1))
    term_5 = gamma*np.transpose(V2).dot(np.transpose(S2)).dot(S2.dot(V2).dot(V1)-W2.dot(Y))
    return term_1+term_2+term_3 + term_5
def SGD_U1(sigma,U1,U2,V1,V2,rating_matrix,alpha,P1,lamda,gamma,S1,X,W1):
    term_1 = (sigma*(U1.dot(U2).dot(V2).dot(V1)-rating_matrix)).dot(np.transpose(U2.dot(V2).dot(V1)))
    term_2 = alpha * np.transpose(P1).dot(P1.dot(U1).dot(U2)-U2).dot(np.transpose(U2))
    term_3 = lamda * U1
    term_4 = gamma*(U1.dot(U2).dot(np.transpose(S1))-np.transpose(X).dot(W1)).dot(S1).dot(np.transpose(U2))
    return term_1+term_2+term_3+term_4
def SGD_V2(U2,U1,sigma,V2,V1,rating_matrix,beta,Q1,m1,lamda,gamma,S2,W2,Y):
    term_1 = np.transpose(U2).dot(np.transpose(U1)).dot(sigma*(U1.dot(U2).dot(V2).dot(V1)-rating_matrix)).dot(np.transpose(V1))
    term_2 = beta*V2.dot(np.eye(m1,m1)-V1.dot(Q1)).dot(np.transpose(np.eye(m1,m1)-V1.dot(Q1)))
    term_3 = lamda*V2
    term_4 = gamma*np.transpose(S2).dot(S2.dot(V2).dot(V1)-W2.dot(Y)).dot(np.transpose(V1))
    return term_1+term_2+term_3+term_4
def SGD_U2(U1,sigma,U2,V2,V1,rating_matrix,beta,gamma,S1,X,W1,alpha,P1,n1,lamda):
    term_1 = np.transpose(U1).dot(sigma*(U1.dot(U2).dot(V2).dot(V1)-rating_matrix)).dot(np.transpose(V1)).dot(np.transpose(V2))
    term_2 = lamda*U2
    term_3 = gamma * np.transpose(U1).dot((U1.dot(U2).dot(np.transpose(S1)))-np.transpose(X).dot(W1)).dot(S1)
    term_4 = alpha*(np.transpose(np.eye(n1,n1)-P1.dot(U1))).dot(np.eye(n1,n1)-P1.dot(U1)).dot(U2)
    return term_1+term_2+term_3+term_4


# in HIRE, the Learning rate is determinated by lipschitz continuity
def Lipschitz_W1(X,corrupted_rate,gamma,z):
    '''
    X: the user features with dimension d_x-by-k
    corrupted_rate: hyper-parameter in MDA
    gamma: hyper-parameter
    z: equal to d_x; the user flat feature dimension
    '''
    term_1 = gamma * X.dot(np.transpose(X))
    term_2 = (1 - corrupted_rate) * (1 - corrupted_rate) * (np.ones([z, z]) - np.diag(np.ones([z]))) * np.dot(X, np.transpose(X))
    term_2 += (1-corrupted_rate) * np.diag(np.ones([z])) * np.dot(X,np.transpose(X))
    return (1/LA.norm(term_1 + term_2, 'fro'))

def Lipschitz_W2(Y,corrupted_rate,gamma,q):
    """
    all parameters described in paper
    q: d_y; equal to item flat feature dimension
    """
    term_1 = gamma * Y.dot(np.transpose(Y))
    term_2 = (1-corrupted_rate) * (1-corrupted_rate) * (np.ones([q,q]) - np.diag(np.ones([q]))) * np.dot(Y, np.transpose(Y))
    term_2 += (1-corrupted_rate) * np.diag(np.ones([q])) * np.dot(Y, np.transpose(Y))
    return (1/LA.norm(term_1 + term_2, 'fro'))

def Lipschitz_S1(U):
    return (1/LA.norm(np.transpose(U).dot(U), 'fro'))

def Lipschitz_S2(V):
    return (1/LA.norm(np.transpose(V).dot(V), 'fro'))

def Lipschitzz_V1(U1,U2,V2,lamda,beta,Q1,S2,gamma):
    """
    U1,U2: two  user hierarchical feature matrix
    V2: item second feature matrix(second level hierarchical structure)
    lambda: hyper-parameter
    beta: control the contribution in item hierarchical structure
    """
    term_1 = LA.norm(np.transpose(V2).dot(np.transpose(U2)).dot(np.transpose(U1)),'fro') * LA.norm(U1.dot(U2).dot(V2),'fro')
    term_2 = lamda
    term_3 = LA.norm(beta*np.transpose(V2).dot(V2),'fro')*LA.norm(Q1.dot(np.transpose(Q1)),'fro')
    term_4 = LA.norm(gamma*np.transpose(V2).dot(np.transpose(S2)).dot(S2).dot(V2),'fro')
    return 1/(term_1+term_2+term_3+term_4)

def Lipschitzz_U1(U2,V2,V1,alpha,lamda,P1,S1,gamma):
    term_1 = LA.norm(U2.dot(V2).dot(V1),'fro') * LA.norm(np.transpose(V1).dot(np.transpose(V2)).dot(np.transpose(U2)),'fro')
    term_2 = LA.norm(alpha*np.transpose(P1).dot(P1),'fro') * LA.norm(U2.dot(np.transpose(U2)),'fro')
    term_3 = lamda
    term_4 = LA.norm(gamma*U2.dot(np.transpose(S1)).dot(S1).dot(np.transpose(U2)),'fro')
    return 1/(term_1+term_2+term_3+term_4)

def Lipschitzz_V2(U1,U2,beta,V1,Q1,lamda,gamma,S2,m1):
    term_1 = LA.norm(np.transpose(U2).dot(np.transpose(U1)),'fro') * LA.norm(U1.dot(U2),'fro') * LA.norm(V1,'fro') * LA.norm(V1.T,'fro')
    term_2 = LA.norm(beta*(np.eye(m1,m1)-V1.dot(Q1)).dot(np.transpose(np.eye(m1,m1)-V1.dot(Q1))),'fro')
    term_3 = lamda
    term_4 = LA.norm(gamma*np.transpose(S2).dot(S2),'fro') * LA.norm(V1.dot(np.transpose(V1)),'fro')
    return 1/(term_1+term_2+term_3+term_4)
def Lipschitzz_U2(U1,V1,V2,lamda,gamma,S1,P1,alpha,n1):
    term_1 = LA.norm(np.transpose(U1),'fro') * LA.norm(U1,'fro') * LA.norm(V2.dot(V1),'fro') * LA.norm(np.transpose(V1).dot(np.transpose(V2)),'fro')
    term_2 = lamda
    term_3 = LA.norm(gamma*np.transpose(U1).dot(U1),'fro') * LA.norm(np.transpose(S1).dot(S1),'fro')
    term_4 = LA.norm(alpha*np.transpose(np.eye(n1,n1)-P1.dot(U1)).dot(np.eye(n1,n1)-P1.dot(U1)),'fro')
    return 1/(term_1+term_2+term_3+term_4)










