import pandas as pd
import numpy as np

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y
    return one_hot_Y




data =  pd.read_csv("train.csv")
data = np.array(data)


number_of_samples, number_per_sample = data.shape


DATA_TRAIN = data[1000:number_of_samples].T

TRAIN_X = DATA_TRAIN[1:number_per_sample]
TRAIN_Y = DATA_TRAIN[0]


print(TRAIN_Y)

one_hota = one_hot(TRAIN_Y)


print(one_hota)

