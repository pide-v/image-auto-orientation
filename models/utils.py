"""
utils.py contains some basic utility functions
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf

"""
Directory at path must have the following structure:
my-images 
		|-- class-1-images
		|-- class-2-images
		|-- ...
		|-- class-n-images

generate_images takes all the images for each class and generates their rotation at 90, 180 and 270 degrees.
This rotated images and the original one are saved in the dest_path in the following way:

	rot-images
		|-- 0
  		|-- 90
    	|-- 180
      	|-- 270
"""

def generate_images(path, dest_path):
	classes = [f for f in os.listdir(path) if not f.startswith('.')]
	print(f'Found {len(classes)} classes: {classes}')

	for folder in classes:
		print(f'### Processing images of class {folder} ###')
		for rot in [0, 90, 180, 270]:
			try:
				os.mkdir(f'{dest_path}/{rot}')
			except FileExistsError:
				print(f'generate-dataset.py: Directory {dest_path}/{folder}_{rot} already exists. Ignoring creation.')
			for im in os.listdir(f'{path}/{folder}'):
				img = Image.open(f'{path}/{folder}/{im}')
				rot_img = img.rotate(rot)
				print(f'Saving image {im} at {rot} degrees')
				rot_img.save(f'{dest_path}/{rot}/{im[:-5]}_{rot}.JPEG')


"""
generate_dataset returns two numpy arrays x, y of shape (N, w, h, c) and (N,).
N = number of samples
w = sample width
h = sample height
c = number of channels (default = 3)

x, y dtype is uint8.

note: since classes are the folder taken in order we have the following:
	class 0 -> 0 degree rotation
	class 1 -> 180 degree rotation
	class 2 -> 270 degree rotation
	class 3 -> 90 degree rotation

"""
def generate_dataset(path, image_size, channels=3):
	data = tf.keras.utils.image_dataset_from_directory(path, batch_size=1, image_size=image_size)
	size = sum(1 for _ in data)
	x = np.zeros((size, image_size[0], image_size[1], channels), dtype=np.uint8)
	y = np.zeros((size), dtype=np.uint8)

	for i, (image, label) in enumerate(data.as_numpy_iterator()):
		x[i] = image[0]
		y[i] = label[0]

	return x, y