from keras.models import Sequential
from tensorflow import keras
import cv2
import numpy as np

img_path = 'data/test/monke.jpg'

model = keras.models.load_model('model.h5')

picture = cv2.imread(img_path)

picture = cv2.resize(picture, (100, 100))

animals = ['dog', 'cat', 'monke']

arr =[]
arr.append(picture)
arr= np.array(arr)
arr= arr/255

label = model.predict(arr)[0]

print(animals[label.argmax()], label.max())
print('Dog: ', label[0], 'Cat: ', label[1], 'Monke:', label[2])