import cv2 as cv
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


model = tf.keras.models.load_model("model")
print("finished loading")
img=cv.imread(f'demo/7.png')[:,:,0]
img = cv.resize(img, (28, 28))
img=np.invert(np.array([img]))
prediction=model.predict(img)
print("----------------")
print("The predicted value is : ", prediction)
print("----------------")