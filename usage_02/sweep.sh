#!/bin/bash
rm -rf nohup.out
nohup wandb agent $1 > nohup.out

