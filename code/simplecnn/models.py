import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set tf log to error only

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout

from tensorflow.keras.regularizers import l2


# first simple model
def build_model_01(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(Conv2D(16, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    return model


# same structure but with some dropout and l2 regularizations
def build_model_02(input_shape, num_classes):
    model=Sequential()
    
    model.add(Input(shape=input_shape))
    
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(256, kernel_size=(3,3), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    
    model.add(Dense(512, activation='relu'), Dropout)
    model.add(Dropout(0.2))
    
    model.add(Dense(128, activation='relu'))
    
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    return model