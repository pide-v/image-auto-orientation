import sys
sys.path.append("/Users/stefa/Desktop/University/AML/image-auto-orientation/code")
import utils
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



def plot_loss(history):
  x_plot = list(range(1,len(history.history["loss"])+1))
  plt.figure()
  plt.title("Loss")
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.plot(x_plot, history.history['loss'])
  plt.plot(x_plot, history.history['val_loss'])
  plt.legend(['Training', 'Validation'])
  plt.show()

def plot_accuracy(history):
  x_plot = list(range(1,len(history.history["accuracy"])+1))
  plt.figure()
  plt.title("Accuracy")
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.plot(x_plot, history.history['accuracy'])
  plt.plot(x_plot, history.history['val_accuracy'])
  plt.legend(['Training', 'Validation'])
  plt.show()  



dataset_path = "../../../cifar10"
x, y = utils.generate_dataset(dataset_path, (32, 32), channels=3)

plt.imshow(x[12])
plt.show()
print(f"Example: {y[:100]}")



x = x.astype('float32')/255
        
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print("\nBuild the model: \n")

#model06
model_01=Sequential()
model_01.add(Input(shape=(32,32,3)))

model_01.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model_01.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

model_01.add(MaxPooling2D(pool_size=(2,2)))
model_01.add(Dropout(0.25))

model_01.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model_01.add(Conv2D(128, kernel_size=(3,3), activation='relu', kernel_regularizer=l2(0.01)))

model_01.add(Flatten())

model_01.add(Dense(128, activation='relu'))
model_01.add(Dropout(0.25))

model_01.add(Dense(num_classes, activation='sigmoid'))

model_01.summary()

optimizer = 'adam'
loss = "binary_crossentropy"
batch_size = 128
epochs = 15

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

start= time.time()
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
total_time = time.time() - start

#SALVA LE COSE

plot_accuracy(model.history)
plot_loss(model.history)

print("After training")
model_score = model.evaluate(x_test, y_test)
print('Test loss:', model_score[0])
print('Test accuracy:', model_score[1])
