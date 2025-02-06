import tensorflow as tf
import numpy as np
from PIL import Image

img_path = 'test-images/mock-dataset'

data = tf.keras.utils.image_dataset_from_directory(img_path, batch_size=4, image_size=(160, 160))
data_iterator = data.as_numpy_iterator()

#Each batch[0] is of shape (batch_size, img_width, img_height, n_channels)
# In this case (4, 160, 160, 3)

#Batch[1] contains the labels associated with images in batch[0]

"""
img = Image.fromarray(batch[0][0].astype(np.uint8), "RGB")
img.show()
"""

#Iterating on 8 batches of 4 images
i = 0
for batch in data_iterator:
	print(f'{i}: {batch[0].shape}')
	i += 1