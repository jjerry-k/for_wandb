#!/bin/bash

for model in 'vgg16' 'vgg19' 'resnet50' 'resnet101' 'resnet152' 'resnet50v2' 'resnet101v2' 'resnet152v2' 'densenet121' 'densenet169' 'densenet201' 'mobilenet' 'mobilenetv2' 'xception' 'inceptionresnetv2' 'inceptionv3' 'nasnetlarge' 'nasnetmobile' 
do
    nohup python main.py --model $model > logs/nohup.out
done