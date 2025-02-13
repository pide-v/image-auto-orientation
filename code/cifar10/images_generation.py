import tensorflow as tf
import os
from PIL import Image
import numpy as np
import sys
sys.path.append('/Users/stefa_ypsvwdy/Desktop/University/image-auto-orientation/code')

import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #set tf log to error only

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

img_path = "../../cifar10/images"
dest_path = "../../cifar10/dataset"

x_all = np.concatenate((x_train, x_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)


num_images_to_save = 10000

# Scegli casualmente `num_images_to_save` indici senza ripetizioni
random_indices = np.random.choice(len(x_all), num_images_to_save, replace=False)

for class_name in class_names:
    os.makedirs(os.path.join(img_path, class_name), exist_ok=True)

for i, idx in enumerate(random_indices):
    img = Image.fromarray(x_all[idx])  # Converte in PIL
    class_label = class_names[y_all[idx][0]]  # Ottiene la classe come stringa
    class_folder = os.path.join(img_path, class_label)
    
    img_filename = os.path.join(class_folder, f"img_{i}.jpg")  # Nome del file
    img.save(img_filename, "JPEG")

print(f"{len(x_all)} immagini salvate in: {img_path}")

utils.generate_images(img_path, dest_path)
