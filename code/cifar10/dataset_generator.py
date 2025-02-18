import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Imposta i percorsi
base_path = os.path.expanduser("~/aml/image-auto-orientation/cifar10")

normal_path = os.path.join(base_path, 'normal')
rotated_path = os.path.join(base_path, 'rotated')

os.makedirs(normal_path, exist_ok=True)
os.makedirs(rotated_path, exist_ok=True)

rotation_angles = [90, 180, 270]

# Carica il dataset MNIST
(x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

# Combina train e test
dataset = np.concatenate((x_train, x_test), axis=0)
num_images = len(dataset)

# Dividi il dataset in due metà
half = num_images // 2
normal_images = dataset[:half]
rotated_images = dataset[half:]

# Salva le immagini normali
for i, img_array in enumerate(normal_images):
    img = Image.fromarray(img_array)
    img.save(os.path.join(normal_path, f'img_{i}.png'), 'PNG')

# Salva le immagini ruotate
for i, img_array in enumerate(rotated_images):
    img = Image.fromarray(img_array)
    rotation_angle = rotation_angles[i % len(rotation_angles)]
    rotated_img = img.rotate(rotation_angle)
    rotated_img.save(os.path.join(rotated_path, f'img_{i}.png'), 'PNG')

print(f'Dataset salvato in {base_path} con {half} immagini normali e {half} immagini ruotate.')
