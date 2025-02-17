import sys
sys.path.append('/home/pide/aml/image-auto-orientation/code/')
import utils as ut

import models as models

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import time

dataset = 'split-dataset'

train_path = f'/home/pide/aml/image-auto-orientation/{dataset}/train'
test_path = f'/home/pide/aml/image-auto-orientation/{dataset}/test'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(320, 320),
    batch_size=8,
    class_mode="sparse",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(320, 320),
    batch_size=8,
    class_mode="sparse",
    subset="validation"
)

optimizer = 'adam'
loss = "sparse_categorical_crossentropy"

model = models.build_model_07((320,320, 3), 2)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

start = time.time()
history = model.fit(train_generator, validation_data=val_generator, epochs=25)
end = time.time()

total_time = end - start

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(320, 320),
    batch_size=16,
    class_mode="sparse",
    shuffle=False
)

test_loss, test_acc = model.evaluate(test_generator, verbose=1)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

ut.save_model_and_metrics(model, model.count_params(), total_time, history, test_acc, dataset, '../../trained-models', 'model07')
