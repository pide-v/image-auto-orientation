import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set tf log to error only

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = '/home/pide/aml/image-auto-orientation/ss-dataset/train'
test_path = '/home/pide/aml/image-auto-orientation/ss-dataset/test'


#exclude the fc layers using include_top=False
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

out = GlobalAveragePooling2D()(base_model.output)
out = Dense(256, activation='relu')(out)
out = Dense(4, activation='softmax')(out)

model = Model(inputs=base_model.input, outputs=out)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

#model.summary()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="sparse",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="sparse",
    subset="validation"
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

model.save("resnet_orientation.h5")  # Salva il modello in formato HDF5

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="sparse",
    shuffle=False
)

test_loss, test_acc = model.evaluate(test_generator, verbose=1)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")