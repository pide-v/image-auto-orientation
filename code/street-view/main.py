import sys
sys.path.append('/home/pide/aml/image-auto-orientation/code/')
import utils as ut

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
import time


train_dataset = '../../datasets/street-view/train'  # La cartella principale con le sottocartelle per ogni classe
test_dataset  = '../../datasets/street-view/test'

# Creazione del generatore con normalizzazione
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)

batch_size = 8
image_size = (224, 224)

train_generator = datagen.flow_from_directory(
    train_dataset,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset="training"
)

# Validation Generator (15%)
val_generator = datagen.flow_from_directory(
    train_dataset,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset="validation"
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dataset,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
)

print(f"Train: {train_generator.samples}, Val: {val_generator.samples}, Test: {test_generator.samples}")


#exclude the fc layers using include_top=False
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

out = GlobalAveragePooling2D()(base_model.output)
out = Dense(256, activation='relu')(out)
out = Dropout(0.25)(out)
out = Dense(1, activation='sigmoid')(out)

model = Model(inputs=base_model.input, outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

#Addestramento del classificatore
start = time.time()
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=["accuracy"])

#Fine-tuning di alcuni layers di resnet
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

end = time.time()

total_time = end - start

test_loss, test_acc = model.evaluate(test_generator, verbose=1)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")


ut.save_model_and_metrics(model, model.count_params(), total_time, history, test_acc, 'streetview', '../../trained-models', 'resnet50-streetview')