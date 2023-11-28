import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

data =  pd.read_csv("wine.csv")
data['quality'] = data['quality'].replace({'bad': 1, 'good': 0})
data = np.array(data)
m, n = data.shape
#np.random.shuffle(data)


data_train = data.T
Y_train = data_train[-1]
X_train = data_train[0:n-1]
X_train = X_train
_,m_train = X_train.shape









W1 = np.random.rand(10, 11) - 0.5
b1 = np.random.rand(10, 1) - 0.5
W2 = np.random.rand(1, 10) - 0.5
b2 = np.random.rand(1) - 0.5


def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    dZ2 = A2 - Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


z1, a1, z2, a2 = forward_prop(W1=W1, b1=b1, W2=W2, b2=b2, X=X_train)

dW1, db1, dW2, db2 = backward_prop(Z1=z1, A1=a1, Z2=z2, A2=a2, W1=W1, W2=W2, X=X_train, Y=Y_train)

print(db2)

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

W1 ,b1, W2, b2 = update_params(W1 ,b1, W2, b2, dW1, db1, dW2, db2, 0.1)

