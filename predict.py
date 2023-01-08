from keras.models import Sequential
from tensorflow import keras
import cv2
import numpy as np
model = keras.models.load_model('model.h5')

img_path = 'zen-bebe.jpeg'
picture = cv2.imread(img_path)

picture = cv2.resize(picture, (100, 100))



arr =[]
arr.append(picture)
arr= np.array(arr)
arr= arr/255
print(arr.shape)



label = model.predict(arr)
if label < 0.5:
    print("DOG") 
    print(1-label)
else :
    if label >=0.5:
        print("CAT")
        print(label)