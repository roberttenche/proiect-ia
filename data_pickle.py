import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
import pickle

cwd = os.getcwd()

animals = ['dog', 'cat', 'monke']

data = []

for category in animals:
    path = os.path.join(cwd, 'data', 'train', category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        label = animals.index(category)
        arr = cv2.imread(img_path)
        new_arr = cv2.resize(arr, (100, 100))
        data.append([new_arr, label])

random.shuffle(data)

images = []
labels = []

for image, label in data:
    images.append(image)
    labels.append(label)

images = np.array(images)
labels = np.array(labels)

images = images / 255


print(labels)
pickle.dump(images, open('images.pkl', 'wb'))
pickle.dump(labels, open('labels.pkl', 'wb'))
