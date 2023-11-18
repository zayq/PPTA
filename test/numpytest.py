import numpy as np


# La function Relu avec numpy
def relu(x):
    return (np.maximum(0, x))


input = np.ones([11])

w = np.ones([11, 11])*-2

print(input, w)

print(relu(np.dot(input, w)))