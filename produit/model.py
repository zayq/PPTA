import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import keras.optimizers

x = pickle.load(open('x.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))

x = x/255
print(x.shape)



model = Sequential()

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, input_shape=x.shape[1:], activation='relu'))

model.add(Dense(2, activation='softmax'))
opt = keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer=opt, loss='categorical_crossentropy function', metrics=['accuracy'])

model.fit(x, y, epochs=5, validation_split=0.1)

model.save(filepath='model')
