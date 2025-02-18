import sys
sys.path.append('/home/pide/aml/image-auto-orientation/code/')
import utils as ut

import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #set tf log to error only

import time

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split

from tensorflow.keras.regularizers import l2

dataset_path = "../../cifar10-dataset"

x, y = ut.generate_dataset(dataset_path, (32, 32), channels=3)
x = x.astype('float32')/255        
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model=Sequential()
model.add(Input(shape=(32,32,3)))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', kernel_regularizer=l2(0.01)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(1, activation='sigmoid'))

model.summary()

optimizer = 'adam'
loss = "binary_crossentropy"
batch_size = 128
epochs = 30

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

start= time.time()
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.15)
total_time = time.time() - start

model_score = model.evaluate(x_test, y_test)
print('Test loss:', model_score[0])
print('Test accuracy:', model_score[1])

ut.save_model_and_metrics(model, model.count_params(), total_time, history, model_score[1], 'cifar10-split', '../../trained-models', 'cifar10-model')