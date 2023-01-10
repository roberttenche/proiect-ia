from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import pickle


images = pickle.load(open('images.pkl', 'rb'))
labels = pickle.load(open('labels.pkl', 'rb'))

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(units=128, input_shape=images.shape[1:], activation='relu'))
model.add(Dense(units=128, input_shape=images.shape[1:], activation='relu'))


model.add(Dense(units=3, activation='sigmoid'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(images, labels, epochs=10, validation_split=0.1, batch_size=64)

model.save('model.h5')
