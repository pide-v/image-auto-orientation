import sys
sys.path.append('/home/pide/aml/image-auto-orientation/code/')
import utils as ut

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import time

train_path = '/home/pide/aml/image-auto-orientation/datasets/cifar10-dataset/train'
test_path = '/home/pide/aml/image-auto-orientation/datasets/cifar10-dataset/test'

# Escludi gli strati fully-connected di ResNet50 (include_top=False)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
#base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
base_model.trainable = False  # Congela il modello di base per la fase iniziale di addestramento

# Aggiungi nuovi strati personalizzati
#out = GlobalAveragePooling2D()(base_model.output)
out = Flatten()(base_model.output)
out = Dense(32, activation='relu')(out)
out = Dense(1, activation='sigmoid')(out)

model = Model(inputs=base_model.input, outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15
)

# Generatore di training
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(32, 32),
    batch_size=16,
    class_mode="binary",
    subset="training"
)

# Generatore di validazione
val_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(32, 32),
    batch_size=16,
    class_mode="binary",
    subset="validation"
)

# Riduzione del learning rate quando la validazione non migliora
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-5
)

# Early stopping per fermare l'addestramento se la validazione non migliora
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)


# Addestramento iniziale senza fine-tuning
start = time.time()
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[lr_scheduler, early_stopping]
)

# Fine-tuning dei layer superiori di ResNet50
for layer in base_model.layers[-20:]:  # Sblocca gli ultimi 20 strati
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9),
              loss='binary_crossentropy',
              metrics=["accuracy"])

# Fine-tuning aggiuntivo
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25,
    callbacks=[lr_scheduler, early_stopping]
)

end = time.time()
total_time = end - start

# Preparazione per il test
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(32, 32),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

# Valutazione del modello sui dati di test
test_loss, test_acc = model.evaluate(test_generator, verbose=1)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Salvataggio del modello e dei risultati
ut.save_model_and_metrics(model, model.count_params(), total_time, history, test_acc, 'sports-dataset', '../../trained-models', 'resnet50-cifar10')
