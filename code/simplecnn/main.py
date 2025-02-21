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
from tensorflow.keras.callbacks import EarlyStopping

import time

dataset = 'street-view'

train_path = f'/home/pide/aml/image-auto-orientation/datasets/{dataset}/train'
test_path = f'/home/pide/aml/image-auto-orientation/datasets/{dataset}/test'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True
)

optimizer = 'adam'
loss = "binary_crossentropy"

model = models.build_model_02((224,224, 3), 1)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

start = time.time()
history = model.fit(train_generator, validation_data=val_generator, epochs=30, callbacks=[early_stopping])
end = time.time()

total_time = end - start

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode="binary",
    shuffle=False
)

test_loss, test_acc = model.evaluate(test_generator, verbose=1)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

ut.save_model_and_metrics(model, model.count_params(), total_time, history, test_acc, dataset, '../../trained-models', 'model03-streetview')
