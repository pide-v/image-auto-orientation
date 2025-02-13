import models as models
from sklearn.model_selection import train_test_split

from tensorflow import keras
import numpy as np

import os

import sys
sys.path.append('/Users/stefa_ypsvwdy/Desktop/University/image-auto-orientation/code')

import train 
import utils

"""
Here you can choose which model to train from the list and then evaluate it
"""

dest_path = "../../data-generation/mock-dataset"

# print("Models list: \nmodel_01 \nmodel_02 \n")
# model = input("Please insert the model you would like to train: ")

x_test_final = []
y_test_final = []
total_time = 0

optimizer = 'adam'
loss = "categorical_crossentropy"

model = models.build_model_01((426,320, 3), 4)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

j = 0
num_chunks = len(os.listdir(f'{dest_path}/npy')) // 2

print("before training\n\n")
while j < 15:
    for i in range(num_chunks):
        
        print(f"Working with the {i}th chunk:\n")
        
        x, y = utils.load_dataset_split(dest_path, i)
        x = x.astype('float32')/255
        y = keras.utils.to_categorical(y, 4)
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)
        x_test_final.append(x_test)
        y_test_final.append(y_test)
        
        total_time += train.train_model(model, 512, 1, x_train, y_train)
        del x, y, x_train, x_test, y_train, y_test
    j += 1
    
print("After training")
model_score = train.evaluate_model(model, x_test_final, y_test_final)
print('Test loss:', model_score[0])
print('Test accuracy:', model_score[1])