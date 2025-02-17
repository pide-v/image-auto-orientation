from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np

test_path = '/home/pide/aml/image-auto-orientation/ss-dataset/test'
model = load_model("resnet_orientation.h5")

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="sparse",
    shuffle=False
)

test_loss = model.evaluate(test_generator, verbose=1)

print(f"Test Loss: {test_loss:.4f}")
#print(f"Test Accuracy: {test_acc:.4f}")

y = model.predict(model)
