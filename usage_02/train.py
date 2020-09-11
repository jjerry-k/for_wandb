import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, utils

from model import *

import wandb
from wandb.keras import WandbCallback

tf.random.set_seed(42)

hyperparameter_defaults = dict(
model_name="mobilenet",
dropout = 0.5,
freeze = 1,
batch_size = 128,
learning_rate = 0.001,
epochs = 5,
GPUs="0"
)

wandb.init(project="usage_02", config=hyperparameter_defaults)
config = wandb.config

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=config.GPUs



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

# Data Prepare

URL = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
path_to_zip  = tf.keras.utils.get_file('flower_photos.tgz', origin=URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'flower_photos')

category_list = [i for i in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, i)) ]
print(category_list)

num_classes = len(category_list)
img_size = 128

def read_img(path, img_size):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (img_size, img_size))
    return img

imgs_tr = []
labs_tr = []

imgs_val = []
labs_val = []

for i, category in enumerate(category_list):
    path = os.path.join(PATH, category)
    imgs_list = os.listdir(path)
    print("Total '%s' images : %d"%(category, len(imgs_list)))
    ratio = int(np.round(0.05 * len(imgs_list)))
    print("%s Images for Training : %d"%(category, len(imgs_list[ratio:])))
    print("%s Images for Validation : %d"%(category, len(imgs_list[:ratio])))
    print("=============================")

    imgs = [read_img(os.path.join(path, img),img_size) for img in imgs_list]
    labs = [i]*len(imgs_list)

    imgs_tr += imgs[ratio:]
    labs_tr += labs[ratio:]
    
    imgs_val += imgs[:ratio]
    labs_val += labs[:ratio]

imgs_tr = np.array(imgs_tr)/255.
labs_tr = utils.to_categorical(np.array(labs_tr), num_classes)

imgs_val = np.array(imgs_val)/255.
labs_val = utils.to_categorical(np.array(labs_val), num_classes)

print(imgs_tr.shape, labs_tr.shape)
print(imgs_val.shape, labs_val.shape)

# Build Network
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = build_model(config, num_classes)
    loss = 'binary_crossentropy' if num_classes==1 else 'categorical_crossentropy'
    model.compile(optimizer=optimizers.Adam(config.learning_rate), loss=loss, metrics=['acc'])

# Training Network

model.fit(x=imgs_tr, y=labs_tr, batch_size=config.batch_size, epochs=config.epochs, 
            callbacks = [WandbCallback()], 
            validation_data=(imgs_val, labs_val))