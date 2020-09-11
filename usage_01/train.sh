#!/bin/bash

for model in 'inceptionv3' 'nasnetlarge' 'nasnetmobile' 
do
    nohup python main.py --model $model > logs/nohup.out
done
