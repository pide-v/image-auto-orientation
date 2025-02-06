import numpy as np
from PIL import Image
import os

#path must be a folder containing only N subfolders. Each subfolder's name represents the
#class of the images that it contains. 

#dest_path specifies where to save the 4 folders a_0, a_90, a_180, a_270 for 
#each subfolder a.

path = 'test-images/images'
dest_path = 'test-images/mock-dataset'

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

		

