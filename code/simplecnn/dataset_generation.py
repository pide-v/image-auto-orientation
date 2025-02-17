import sys
sys.path.append('/home/pide/aml/image-auto-orientation/code/')

import utils

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set tf log to error only

train_path = '/home/pide/aml/image-auto-orientation/images/imagenette2-320/train'
test_path = '/home/pide/aml/image-auto-orientation/images/imagenette2-320/val'

#generate dataset with 50% of images at 0 and the other 50% at a random rotation
utils.generate_images_diff(train_path, '/home/pide/aml/image-auto-orientation/split-dataset/train')
#utils.generate_images_diff(test_path, '/home/pide/aml/image-auto-orientation/split-dataset/test')

#generate dataset with 50% of images at 0 and the other 50% at a random rotation. then 15% of images in 0 are copied in 
#0 with a random rotation 
#utils.generate_images_dupl(train_path, '/home/pide/aml/image-auto-orientation/dupl-dataset/train', 0.15)
#utils.generate_images_dupl(test_path, '/home/pide/aml/image-auto-orientation/dupl-dataset/test', 0.15)

#generate dataset with all images at 0 and all images at a random rotation
utils.generate_images_full(train_path, '/home/pide/aml/image-auto-orientation/full-dataset/train')
utils.generate_images_full(test_path, '/home/pide/aml/image-auto-orientation/full-dataset/test')


