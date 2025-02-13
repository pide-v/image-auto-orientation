import sys
sys.path.append('/Users/stefa_ypsvwdy/Desktop/University/image-auto-orientation/code')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set tf log to error only

import utils

'''
Run this file to generate dataset in the folder mock-dataset and to save in the folder .npy the numpy arrays
'''

img_path = "../../data-generation/images"
dest_path = "../../data-generation/mock-dataset"

utils.generate_images(img_path, dest_path)
utils.generate_dataset_splits(dest_path, (426,320), channels=3, chunk_size=5000)