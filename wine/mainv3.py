import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('wine.csv')
data['quality'] = data['quality'].replace({'bad': 1, 'good': 0})
data = np.array(data)
np.random.shuffle(data)
m, n =  data.shape


training_data = data.T




class NeuralNetwork():
    
    def __init__(self):
        
        self.w1 = np.random.randn(11, 11) * np.sqrt(2 / 11)  
        self.b1 = np.random.randn(11, 1) * np.sqrt(2 / 11)
        self.w2 = np.random.randn(11, 1) * np.sqrt(2 / 11)
        self.b2 = np.random.randn(11, 1) * np.sqrt(2 / 11)
        
        self.m, self.n =  data.shape
        self.loss_history = []
        self.accuracy_history = []
        self.training_outputs = training_data[-1]

        self. training_inputs = training_data[0:n - 1]
        mean = np.mean(self.training_inputs, axis=1, keepdims=True)
        std = np.std(self.training_inputs, axis=1, keepdims=True)
        self.training_inputs = (self.training_inputs - mean) / std
        self.iterations = 100000000
        self.a = 0.001
        np.random.seed(0)
    def Loss(self, a2, y):
        loss = -np.mean(y * np.log(a2) + (1 - y) * np.log(1 - a2))
        return loss    
        
    def Train(self):
        
        for x in range(self.iterations):
            z1, a1, z2, a2 = self.ForWardPropagation(self.training_inputs)
            dw1, db1, dw2, db2 = self.BackWardPropagation(z1, a1, a2, self.w2, self.training_inputs, self.training_outputs)
            self.w1, self.b1, self.w2, self.b2 = self.UpdateParams(self.w1, self.b1, self.w2, self.b2, dw1, db1, dw2, db2, self.a)
            loss = self.Loss(a2, self.training_outputs)
            self.loss_history.append(loss)
            predictions = self.Predict(a2)

            self.accuracy_history.append(self.Accuracy(predictions, self.training_outputs))
            
            
            if x % 1000 == 0:
                print("Iteration: ", x)
                predictions = self.Predict(a2)

                print(self.Accuracy(predictions, self.training_outputs))
            
        

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, int(Y.max()) + 1))
        one_hot_Y[np.arange(Y.size), Y.astype(int)] = 1
        return one_hot_Y.T
    def Accuracy(self, p, y):
        return np.sum(p == y) / y.size
        
    def Predict(self, a2):
        return np.argmax(a2, 0)
    
    
    def ReLU(self, x):
        return (np.maximum(0, x))
    
    
    def ReLUDerivative(self, x):
        return x > 0
    
    
    def OutputActivation(self, x):
        return 1 / (1 + np.exp(-x))

          
    def ForWardPropagation(self, inputs):
        
        z1 = self.w1.dot(inputs) + self.b1
        a1 = self.ReLU(z1)
        z2 = self.w2.dot(a1) + self.b2 
        a2 = self.OutputActivation(z2)
        print(a2)
        
        return z1, a1, z2, a2

    def BackWardPropagation(self, z1, a1, a2, w2, x, y):
        one_hot_Y = self.one_hot(y)
        dz2 = a2 - one_hot_Y
        dw2 = 1 / self.m * dz2.dot(a1.T)
        db2 = 1 / self.m * np.sum(dz2)
        
        dz1 = w2.T.dot(dz2) * self.ReLUDerivative(z1)
        dw1 = 1 / self.m * dz1.dot(x.T)
        db1 = 1 / self.m * np.sum(dz1)
        return dw1, db1, dw2, db2
    
    def UpdateParams(self, W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        W1 = W1 - (alpha * dW1)
        b1 = b1 - (alpha * db1)    
        W2 = W2 - (alpha * dW2)  
        b2 = b2 - (alpha * db2)   
        return W1, b1, W2, b2


NN = NeuralNetwork()

NN.Train()
plt.plot(NN.accuracy_history)
plt.title('Performance de Mon model')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.show()