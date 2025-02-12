"""
utils.py contains some basic utility functions
"""

import os
import shutil
import random
import numpy as np
from PIL import Image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set tf log to error only
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
	print(f'utils/generate_images: Found {len(classes)} classes: {classes}')


	for rot in [0, 90, 180, 270]:
		try:
			os.mkdir(f'{dest_path}/{rot}')
		except FileExistsError:
			print(f'utils/generate_images: Directory {dest_path}/{rot} already exists. Make sure the destination folder is empty before starting. Aborting...')
			break
		for folder in classes:
			print(f'utils/generate_images: Saving images of class {folder} at {rot} degrees...')
			for im in os.listdir(f'{path}/{folder}'):
				img = Image.open(f'{path}/{folder}/{im}')
				rot_img = img.rotate(rot)
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

"""
generate_dataset_splits divides the dataset in chunks and saves two .npy files for each chunk.
(one for sample and one for labels). The directory structure is the following:

	dataset
		|-- 0
		|-- 90
		|-- 180
		|-- 270
		|-- npy
			 |-- x0.npy
			 |-- y0.npy
			 |-- ...
			 |-- xn.npy
			 |-- yn.npy

	make sure to call this function once 0, 90, 180, 270 have been generated. folder /npy
	is automatically generated.

	@param chunk_size defines the number of samples for each chunk.

"""
def generate_dataset_splits(path, image_size, channels=3, chunk_size=10000):
	name = 'utils/generate_dataset_splits'

	data = tf.keras.utils.image_dataset_from_directory(path, batch_size=1, image_size=image_size)
	data_iter = data.as_numpy_iterator()
	size = sum(1 for _ in data)

	print(f'{name}: Saving {size} samples.')

	data = tf.keras.utils.image_dataset_from_directory(path, batch_size=1, image_size=image_size, shuffle=True)

	n_chunks = size // chunk_size
	exc_samples = size % chunk_size

	os.makedirs(f"{path}/npy", exist_ok=True)
	print(f'{name}: Saving {n_chunks} chunks of size {chunk_size}.')
	for i in range(n_chunks):
		x = np.zeros((chunk_size, image_size[0], image_size[1], channels), dtype=np.uint8)
		y = np.zeros((chunk_size), dtype=np.uint8)
		for j in range(chunk_size):
			try:
				sample = next(data_iter)
			except StopIteration:
				break
			x[j] = sample[0][0]
			y[j] = sample[1][0]

		with open(f'{path}/npy/x{i}.npy', 'wb') as f:
			np.save(f, x)
		with open(f'{path}/npy/y{i}.npy', 'wb') as f:
			np.save(f, y)

	if exc_samples != 0:
		print(f'{name}: Saving {exc_samples} exceeding samples.')
		x = np.zeros((exc_samples, image_size[0], image_size[1], channels), dtype=np.uint8)
		y = np.zeros((exc_samples), dtype=np.uint8)
		for i in range(exc_samples):
			try:
				sample = next(data_iter)
			except StopIteration:
				break
			x[i] = sample[0][0]
			y[i] = sample[1][0]
		with open(f'{path}/npy/x{n_chunks}.npy', 'wb') as f:
			np.save(f, x)
		with open(f'{path}/npy/y{n_chunks}.npy', 'wb') as f:
			np.save(f, y)


"""
function load_dataset_split return two numpy arrays x, y from the splits previously
saved using generate_dataset_splits. 

@param split defines which split to return. split=4 means that x4 and y4 will be loaded.

@example
for i in range(len(os.listdir(f'{path}/npy'))//2):
	x, y = ut.load_dataset_split(path, i)
	# Training ...

this is a convenient way of using this function.

"""
def load_dataset_split(path, split):
	try:
		print(f'utils/load_dataset_split: Loading split {split}...')
		x = np.load(f'{path}/npy/x{split}.npy')
		y = np.load(f'{path}/npy/y{split}.npy')
		return x, y
	except FileNotFoundError:
		print(name, 'utils/load_dataset_split: Split does not exists. Check indexs')


"""
Useless function. solution to a problem that never existed :/
"""

def dataset_from_directory(path, dest_path=0):
	#given n classes
	#create temp directory with n folders
	#for each image in original folder, copy it to a random temp folder
	#generate_images on temp folder
	#delete temp folder
	#generate npy as usual
	if dest_path == 0:
		os.mkdir(f'{path}/dataset')

	classes = [f for f in os.listdir(path) if not f.startswith('.')]
	images = []

	os.mkdir(f'{path}/temp')
	for folder in classes:
		os.mkdir(f'{path}/temp/{folder}')

	for folder in classes:
		for image in os.listdir(f'{path}/{folder}'):
			dest_folder = random.choice(classes)
			shutil.copy(f'{path}/{folder}/{image}', f'{path}/temp/{dest_folder}')

	generate_images(f'{path}/temp', dest_path)
	generate_dataset_splits(dest_path, (213, 160), chunk_size=10)

	shutil.rmtree(f'{path}/temp')

