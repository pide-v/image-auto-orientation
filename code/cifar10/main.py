import sys
sys.path.append('/Users/stefa_ypsvwdy/Desktop/University/image-auto-orientation/code')
import utils
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #set tf log to error only

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

def plot_accuracy(history):
  x_plot = list(range(1,len(history.history["accuracy"])+1))
  plt.figure()
  plt.title("Accuracy")
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.plot(x_plot, history.history['accuracy'])
  plt.plot(x_plot, history.history['val_accuracy'])
  plt.legend(['Training', 'Validation'])
  




dataset_path = "../../cifar10/dataset"
x, y = utils.generate_dataset(dataset_path, (32,32), channels=3)

plt.imshow(x[10], cmap='gray')
plt.show()
print(f"Example: {y[10]}")

x = x.astype('float32')/255
y = keras.utils.to_categorical(y, 4)
        
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print("Build the model: \n")

model = Sequential()
model.add(Input(shape=(32,32,3)))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu',  kernel_regularizer=l2(0.01)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))

model.summary()

optimizer = 'adam'
loss = "categorical_crossentropy"
batch_size = 128
epochs = 20

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)


plot_accuracy(model.history)
plot_loss(model.history)

print("After training")
model_score = model.evaluate(x_test, y_test)
print('Test loss:', model_score[0])
print('Test accuracy:', model_score[1])

