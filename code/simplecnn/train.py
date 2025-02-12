import tensorflow as tf
import sys

import numpy  as np
import matplotlib.pyplot as plt

from time import time


def plot_loss(history):
  x_plot = list(range(1,len(history.history["loss"])+1))
  plt.figure()
  plt.title("Loss")
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.plot(x_plot, history.history['loss'])
  plt.plot(x_plot, history.history['val_loss'])
  plt.legend(['Training', 'Validation'])

def plot_accuracy(history):
  x_plot = list(range(1,len(history.history["accuracy"])+1))
  plt.figure()
  plt.title("Accuracy")
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.plot(x_plot, history.history['accuracy'])
  plt.plot(x_plot, history.history['val_accuracy'])
  plt.legend(['Training', 'Validation'])
  
  
dataset_path = "Desktop/data-generation/mock-dataset"

image_size = (426,320)
channels = 3

batch_size = 128
epochs = 50

def train_model(model, batch_size, epochs, x_train, y_train):  
  start = time()
  model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
  total_time = start - time()
  return total_time

def evaluate_model(model, x_test, y_test):   
  plot_loss(model.history)
  plot_accuracy(model.history)

  score = model.evaluate(x_test, y_test)

  return score

