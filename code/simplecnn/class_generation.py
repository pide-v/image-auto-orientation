import sys
sys.path.append('/Users/stefa_ypsvwdy/Desktop/University/image-auto-orientation/code')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set tf log to error only

import utils as utils

'''
Run this file to generate dataset in the folder mock-dataset
'''

img_path = "Desktop/data-generation/images"
dest_path = "Desktop/data-generation/mock-dataset"

utils.generate_images(img_path, dest_path)