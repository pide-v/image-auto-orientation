import sys
sys.path.append('/home/pide/aml/image-auto-orientation/code/')
import utils as ut

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import time

train_path = '/home/pide/aml/image-auto-orientation/datasets/sports/train'
test_path = '/home/pide/aml/image-auto-orientation/datasets/sports/test'

# Escludi gli strati fully-connected di ResNet50 (include_top=False)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Congela il modello di base per la fase iniziale di addestramento

# Aggiungi nuovi strati personalizzati
out = GlobalAveragePooling2D()(base_model.output)
#out = Dense(512, activation='relu')(out)
out = Dense(1, activation='sigmoid')(out)

model = Model(inputs=base_model.input, outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Data augmentation per migliorare la generalizzazione
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,  # Aggiungi rotazioni randomiche
    width_shift_range=0.2,  # Traslazione orizzontale
    height_shift_range=0.2,  # Traslazione verticale
    shear_range=0.2,  # Trasformazioni geometriche
    zoom_range=0.2,  # Zoom
    horizontal_flip=True,  # Flip orizzontale
    fill_mode='nearest',
    validation_split=0.15  # Utilizzo di una porzione del dataset per la validazione
)

# Generatore di training
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=8,
    class_mode="binary",
    subset="training"
)

# Generatore di validazione
val_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=8,
    class_mode="binary",
    subset="validation"
)

# Riduzione del learning rate quando la validazione non migliora
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
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
    epochs=10,
    callbacks=[lr_scheduler, early_stopping]
)

# Fine-tuning dei layer superiori di ResNet50
for layer in base_model.layers[-20:]:  # Sblocca gli ultimi 20 strati
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Usa un learning rate più basso
              loss='binary_crossentropy',
              metrics=["accuracy"])

# Fine-tuning aggiuntivo
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    callbacks=[lr_scheduler, early_stopping]
)

end = time.time()
total_time = end - start

# Preparazione per il test
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(320, 320),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

# Valutazione del modello sui dati di test
test_loss, test_acc = model.evaluate(test_generator, verbose=1)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Salvataggio del modello e dei risultati
ut.save_model_and_metrics(model, model.count_params(), total_time, history, test_acc, 'sports-dataset', '../../trained-models', 'resnet50-fc-sports')
