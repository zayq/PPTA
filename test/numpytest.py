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

#print(w)

a = [1, 2, 3]
b = [2, 3, 4]
a = np.array([a])
b = np.array([b])
#print(np.dot(a, b))





a = np.array([1, 0, 3, 9, 2, 2])
temp = []


for i, element in enumerate(a):
    
    if element != a.max():
        temp.append(0)

    else:
        temp.append(1)

b = np.array(temp)

print(b)