import utils

img_path = 'data-generation/test-images/images'
dest_path = 'data-generation/test-images/mock-dataset'

utils.generate_images(img_path, dest_path)
data_iterator = utils.generate_dataset(dest_path)

batch = data_iterator.next()

print(batch[0].shape)
print(batch[1]) #labels