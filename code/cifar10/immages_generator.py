import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Imposta i percorsi
save_path = os.path.expanduser("../../cifar10/dataset")
os.makedirs(save_path, exist_ok=True)

# Crea le cartelle per le rotazioni
rotations = [0, 90, 180, 270]
for rot in rotations:
    os.makedirs(os.path.join(save_path, str(rot)), exist_ok=True)

# Carica il dataset CIFAR-10
(x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

# Combina train e test
x_all = np.concatenate((x_train, x_test), axis=0)

# Seleziona 10.000 immagini casualmente
num_images_to_save = 10000
random_indices = np.random.choice(len(x_all), num_images_to_save, replace=False)
selected_images = x_all[random_indices]

# Salva le immagini ruotate nelle rispettive cartelle
for i, img_array in enumerate(selected_images):
    img = Image.fromarray(img_array)
    for rot in rotations:
        rotated_img = img.rotate(rot)
        img_filename = os.path.join(save_path, str(rot), f"img_{i}.jpg")
        rotated_img.save(img_filename, "JPEG")

print(f"✅ Salvate {num_images_to_save * 4} immagini in {save_path}, divise per rotazione.")
