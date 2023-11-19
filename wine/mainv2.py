from typing import List
from getData import readCSV
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return (np.maximum(0, x))


def sigmoidDerivative(dy, x):
    sig = sigmoid(x)
    return dy * sig * (1 - sig)

def reluDerivative(dy, x):
    dZ = np.array(dy, copy = True)
    dZ[x <= 0] = 0
    return dZ


wines = readCSV()

GOOD = 0
BAD = 1
NUMBER_INPUTS = 11


data = []

for wine in wines:
    
    quality = None
    quality = GOOD if wine['quality'] == 'good' else BAD
    parameters = []
    for parameter in list(wine.values()):
        if isinstance(parameter, (int, float)):
            parameters.append(parameter)
    new_data = [parameters, quality]
    data.append(new_data)
            
np.random.seed(0)
inputs = np.array([item[0] for item in data])
outputs = np.array([item[1] for item in data])


w_i_h = np.random.randn(NUMBER_INPUTS, NUMBER_INPUTS) *0.5
w_h_h = np.random.randn(NUMBER_INPUTS, NUMBER_INPUTS) *0.5
w_h_o = np.random.randn(NUMBER_INPUTS) *0.5


ARCHITECTURE = [
    {"inputs": NUMBER_INPUTS, "outputs": NUMBER_INPUTS, "activation": "relu", "w": w_i_h},
    {"inputs": NUMBER_INPUTS, "outputs": NUMBER_INPUTS, "activation": "relu", "w": w_h_h},
    {"inputs": NUMBER_INPUTS, "outputs": NUMBER_INPUTS, "activation": "relu", "w": w_h_o},
    {"inputs": NUMBER_INPUTS, "outputs": 1, "activation": "sigmoid"}
]

bs = np.random.randn(len(ARCHITECTURE) - 1) *0.5
    
def CalculateLayerOutput(inputs, w, b):
    
    outputs_no_activation = np.dot(inputs, w) + b
    
    outputs = relu(outputs_no_activation)
    
    return outputs


def CalculateFullOutput(base_inputs, w_i_h, w_h_h, w_h_o, b):
    
    current_inputs = base_inputs
    
    for i, layer in enumerate(ARCHITECTURE):
        if "w" not in layer:
            return sigmoid(current_inputs)
        
        weights = None
        
        if i == 0:
            weights = w_i_h
        if i == 1:
            weights = w_h_h
        if i == 2:
            weights = w_h_o
            
        outputs = CalculateLayerOutput(inputs=current_inputs, w=weights, b=b)
        current_inputs = outputs


def CostFunction(w_i_h, w_h_h, w_h_o, b):
    
    
    total_cost = 0
    for i, input in enumerate(inputs):
        output = CalculateFullOutput(input, w_i_h, w_h_h, w_h_o, b=b)
        difference = output - outputs[i]
        cost = difference*difference
        total_cost += cost
    
    average_cost = total_cost / len(inputs)
    
    return average_cost
    
def Train(bs, w_i_h, w_h_h, w_h_o):
    rate = 0.5
    
    for x in range(10000):
        for i, layer in enumerate(ARCHITECTURE):
            
            
            if "w" in layer and i<3:
                cost = CostFunction(w_i_h, w_h_h, w_h_o, bs[i])
                    
                if i == 0:
                    weights = w_i_h
                    weights += 0.01
                    
                    wslope = (CostFunction(weights, w_h_h, w_h_o, bs[i]) - cost) / 0.01
                    w_i_h -= wslope * rate
                elif i == 1:
                    weights = w_h_h
                    weights += 0.01
                    
                    wslope = (CostFunction(w_i_h, weights, w_h_o, bs[i]) - cost) / 0.01
                    w_h_h -= wslope * rate
                elif i == 2:
                    weights = w_h_o
                    weights += 0.01
                    
                    wslope = (CostFunction(w_i_h, w_h_h, weights, bs[i]) - cost) / 0.01
                    w_h_o -= wslope * rate
                
                    

                b = bs[i] + 0.01

                bslope = (CostFunction(weights, w_h_h, w_h_o, b) - cost) / 0.01
                bs[i] -= np.sum(bslope) * rate
    print(CostFunction(w_i_h, w_h_h, w_h_o, bs[i]))



                    
            
            
               
                
                



Train(bs=bs, w_i_h=w_i_h, w_h_h=w_h_h, w_h_o=w_h_o)
#for input in inputs:
#print(CalculateFullOutput(input))
#print(CalculateLayerOutput(inputs[0], w_h_o, 2, 'sigmoid'))
    
    


