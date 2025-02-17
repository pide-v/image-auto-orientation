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

train_path = '/home/pide/aml/image-auto-orientation/full-dataset/train'
test_path = '/home/pide/aml/image-auto-orientation/full-dataset/test'

#exclude the fc layers using include_top=False
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(320, 320, 3))
base_model.trainable = False

out = GlobalAveragePooling2D()(base_model.output)
out = Dense(256, activation='relu')(out)
out = Dropout(0.25)(out)
out = Dense(1, activation='sigmoid')(out)

model = Model(inputs=base_model.input, outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(320, 320),
    batch_size=8,
    class_mode="binary",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(320, 320),
    batch_size=8,
    class_mode="binary",
    subset="validation"
)

#Addestramento del classificatore
start = time.time()
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15
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
    epochs=15
)

end = time.time()

total_time = end - start

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(320, 320),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

test_loss, test_acc = model.evaluate(test_generator, verbose=1)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")


ut.save_model_and_metrics(model, model.count_params(), total_time, history, test_acc, 'full-dataset', '../../trained-models', 'resnet50-fc')