import sys
sys.path.append("/Users/stefa/Desktop/University/AML/image-auto-orientation/code")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set tf log to error only

import utils

'''
Run this file to generate dataset in the folder mock-dataset and to save in the folder .npy the numpy arrays
'''

img_path_train = "../../../data-generation/images/train"
dest_path_train = "../../../data-generation/dataset/train"

img_path_test = "../../../data-generation/images/test"
dest_path_test = "../../../data-generation/dataset/test"

#utils.generate_images(img_path_train, dest_path_train)

utils.generate_images(img_path_test, dest_path_test)


#utils.generate_dataset_splits(dest_path, (426,320), channels=3, chunk_size=5000)