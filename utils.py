import os
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
		|-- class-1-images_0
		|-- class-1-images_90
		|-- class-1-images_180
		|-- class-1-images_270
		|-- class-2-images_0
		|-- class-2-images_90
		|-- ...
		|-- class-n-images_270
"""

def generate_images(path, dest_path):
	classes = [f for f in os.listdir(path) if not f.startswith('.')]
	print(f'Found {len(classes)} classes: {classes}')

	for folder in classes:
		print(f'### Processing images of class {folder} ###')
		for rot in [0, 90, 180, 270]:
			try:
				os.mkdir(f'{dest_path}/{folder}_{rot}')
			except FileExistsError:
				print(f'generate-dataset.py: Directory {dest_path}/{folder}_{rot} already exists. Ignoring creation.')
			for im in os.listdir(f'{path}/{folder}'):
				img = Image.open(f'{path}/{folder}/{im}')
				rot_img = img.rotate(rot)
				print(f'Saving image {im} at {rot} degrees')
				rot_img.save(f'{dest_path}/{folder}_{rot}/{im[:-5]}_{rot}.JPEG')


"""
generate_dataset returns an iterable containing batch_size images as numpy arrays and the respective labels.

@example
	iter = generate_dataset(path)
	batch = iter.next()
	batch[0] --> images
	batch[1] --> labels


"""
def generate_dataset(path):
	data = tf.keras.utils.image_dataset_from_directory(path, batch_size=4, image_size=(160, 160))
	return data.as_numpy_iterator()