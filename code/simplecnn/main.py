import models as models

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tensorflow import keras
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

import sys
sys.path.append("/Users/stefa/Desktop/University/AML/image-auto-orientation/code")

train_path = "../../../data-generation/dataset/train"

test_path = "../../../data-generation/dataset/test"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(320, 320),
    batch_size=128,
    class_mode="sparse",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(320, 320),
    batch_size=128,
    class_mode="sparse",
    subset="validation"
)

optimizer = 'adam'
loss = "sparse_categorical_crossentropy"

model = models.build_model_03((320,320, 3), 4)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

history = model.fit(train_generator, validation_data=val_generator, epochs=10)

model.save("resnet_orientation.h5")

models.plot_accuracy(history)
models.plot_loss(history)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(320, 320),
    batch_size=128,
    class_mode="sparse",
    shuffle=False
)

test_loss, test_acc = model.evaluate(test_generator, verbose=1)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")