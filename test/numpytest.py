import numpy as np


# La function Relu avec numpy
def relu(x):
    return (np.maximum(0, x))

# Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

input = np.ones([11])

w = np.ones([11, 11])*-2


w = np.random.randn(11, 11) *0.1

print(w)

