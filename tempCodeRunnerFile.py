import sys
sys.path.append('/home/pide/aml/image-auto-orientation/code/')
import utils as ut

train = 'images/tiny-imagenet-200/train'
test = 'images/tiny-imagenet-200/test'

dest_train = 'datasets/tiny-imagenet/train'
dest_test = 'datasets/tiny-imagenet/test'

ut.generate_images(train, dest_train)
ut.generate_images(test, dest_test)