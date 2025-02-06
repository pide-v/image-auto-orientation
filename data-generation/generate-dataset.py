import numpy as np
from PIL import Image
import os
import csv


path = 'test-images/imagenette2-160/train'
test_path = 'test-images/test'

classes = [f for f in os.listdir(test_path) if not f.startswith('.')]
print(f'Found {len(classes)} classes: {classes}')

for folder in classes:
	print(f'### Processing images of class {folder} ###')
	for rot in [90, 180, 270]:
		os.mkdir(f'{test_path}/{folder}_{rot}')
		for im in os.listdir(f'{test_path}/{folder}'):
			img = Image.open(f'{test_path}/{folder}/{im}')
			rot_img = img.rotate(rot)
			print(f'Saving image {im} at {rot} degrees')
			rot_img.save(f'{test_path}/{folder}_{rot}/{im[:-5]}_{rot}.JPEG')

		

