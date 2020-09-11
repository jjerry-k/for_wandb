import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, utils, datasets

# Network Building
def build_model():
    cnn = models.Sequential()
    cnn.add(layers.Conv2D(16, 3, activation='relu', input_shape=(28, 28, 1,)))
    cnn.add(layers.MaxPool2D())
    cnn.add(layers.Conv2D(32, 3, activation='relu'))
    cnn.add(layers.MaxPool2D())
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(10, activation='softmax'))
    return cnn