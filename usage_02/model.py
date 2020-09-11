import tensorflow as tf
from tensorflow.keras import models, layers, losses, optimizers
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import VGG16, VGG19, Xception, InceptionResNetV2, InceptionV3
from tensorflow.keras.applications import MobileNet, MobileNetV2, NASNetLarge, NASNetMobile
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2

model_dict = {
    "vgg16": VGG16, 
    "vgg19": VGG19,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "resnet50v2": ResNet50V2,
    "resnet101v2": ResNet101V2,
    "resnet152v2": ResNet152V2,
    "densenet121": DenseNet121,
    "densenet169": DenseNet169,
    "densenet201": DenseNet201,
    "mobilenet": MobileNet,
    "mobilenetv2": MobileNetV2,
    "xception": Xception,
    "inceptionresnetv2": InceptionResNetV2,
    "inceptionv3": InceptionV3, 
    "nasnetlarge": NASNetLarge,
    "nasnetmobile": NASNetMobile    
}

def build_model(config, num_classes=10, name="model"):
    assert config.model_name.lower() in model_dict.keys(), f"Please, check pretrained model list {list(model_dict.keys())}"

    last_activation = "softmax" if num_classes > 1 else "sigmoid"

    
    base_model = model_dict[config.model_name.lower()](include_top=False, weights="imagenet", pooling="avg")

    if config.freeze:
        base_model.trainable = False
    
    output = layers.Dropout(config.dropout, name=f"{name}_dropout")(base_model.output)
    output = layers.Dense(num_classes, last_activation, name=f"{name}_output")(output)

    model = models.Model(base_model.input, output, name=name)
    return model