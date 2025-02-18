import sys
sys.path.append('/home/pide/aml/image-auto-orientation/code/')
import utils as ut

train = 'images/sports/train'
test = 'images/sports/test'

dest_train = 'datasets/sports/train'
dest_test = 'datasets/sports/test'

ut.generate_images_diff(train, dest_train)
ut.generate_images_diff(test, dest_test)