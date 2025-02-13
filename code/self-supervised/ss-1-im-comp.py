"""

ss-1-im-comp

model trained on image completion pretext task.

"""

import sys
sys.path.append('/home/pide/aml/image-auto-orientation/models/')


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set tf log to error only

import tensorflow as tf

import utils as ut
import numpy as np

from sklearn.model_selection import train_test_split

ut.generate_images('/home/pide/aml/image-auto-orientation/images/single-class', 
	'/home/pide/aml/image-auto-orientation/ss-dataset')

### DATA PREPARATION ###
path = '/home/pide/aml/image-auto-orientation/ss-dataset'
x,y = ut.generate_dataset(path, (213, 160))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=77)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
print(f'Training set shape: {x_train.shape}')
print(f'Test set shape: {x_test.shape}')

