import tensorflow as tf
import sys
from sklearn.model_selection import train_test_split

import numpy  as np
import matplotlib.pyplot as plt

from time import time
from utils import generate_dataset


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

print("\n\n\nprima del generate dataset\n\n\n")
x, y = generate_dataset(dataset_path, image_size, channels)
print("\n\n\ndopo del generate dataset\n\n\n")
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
print("\n\n\ndopo lo split\n\n\n")

'''
def train_model(model, optimizer, loss, batch_size, epochs, x_train, y_train, x_, y_test):
    start = time()
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    total_time = start - time()

    plot_loss(model.history)
    plot_accuracy(model.history)

    score = model.evaluate(x_test, y_test)

    return score, total_time


'''

'''
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python train.py nome_modello")
        sys.exit(1)
    
    model_name = sys.argv[1]

    # Import dinamico del modello
    try:
        modello = __import__(f"modelli.{model_name}", fromlist=["build_model1"])
        train_model(modello.build_model1)
    except ModuleNotFoundError:
        print(f"Errore: il modello '{model_name}' non esiste.")
'''

