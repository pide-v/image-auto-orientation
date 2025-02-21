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
"""
dataset_path = "../../mnist-dataset"
x, y = ut.generate_dataset(dataset_path, (28, 28), channels=1)

plt.imshow(x[12], cmap='gray')
plt.show()
print(f"Example: {y[:100]}")



x = x.astype('float32')/255
        
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print("\nBuild the model: \n")
"""
model = Sequential()
model.add(Input(shape=(28,28,1)))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

model.summary()

optimizer = 'adam'
loss = "binary_crossentropy"
batch_size = 128
epochs = 15

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

start= time.time()
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
total_time = time.time() - start


print("After training")
model_score = model.evaluate(x_test, y_test)
print('Test loss:', model_score[0])
print('Test accuracy:', model_score[1])


#SALVA LE COSE
ut.save_model_and_metrics(model, model.count_params(), total_time, history, model_score[1], 'mnist-dataset', '../../trained-models', 'mnist-model')