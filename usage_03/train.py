import os
import sys
import random
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, datasets

import wandb
from wandb.keras import WandbCallback

PYTHONHASHSEED=777
random.seed(777)
np.random.seed(777)
tf.random.set_seed(777)

EPOCHS=10
BATCH_SIZE=1024
# For Efficiency
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Data Loading
(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
train_x, test_x = np.expand_dims(train_x/255., -1), np.expand_dims(test_x/255., -1)

print("Train Data's Shape : ", train_x.shape, train_y.shape)
print("Test Data's Shape : ", test_x.shape, test_y.shape)


wandb.init(project="usage_03", name=f"ALL_{sys.argv[1]}")

cnn = models.Sequential()
cnn.add(layers.Conv2D(16, 3, activation='relu', input_shape=(28, 28, 1,)))
cnn.add(layers.MaxPool2D())
cnn.add(layers.Conv2D(32, 3, activation='relu'))
cnn.add(layers.MaxPool2D())
cnn.add(layers.Flatten())
cnn.add(layers.Dense(10, activation='softmax'))

print("Network Built!")

# Compiling
cnn.compile(optimizer=optimizers.Adam(), 
            loss=losses.sparse_categorical_crossentropy, 
            metrics=['accuracy'])

# Training
history = cnn.fit(train_x, train_y, 
                    epochs=EPOCHS, batch_size=BATCH_SIZE, 
                    validation_data=(test_x, test_y), 
                    callbacks=[WandbCallback()])