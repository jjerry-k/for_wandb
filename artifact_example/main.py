import time
import os
import shutil
import wandb
import random
import torch
from model import *

def save_checkpoint(state, is_best, checkpoint):
    filepath = os.path.join(checkpoint, f"epoch{state['epoch']:03d}.pth.tar")
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.makedirs(checkpoint, exist_ok=True)
    
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "best.pth.tar"))

def data_version_control(config):
    with wandb.init(entity="jjerry", job_type="dvc", project="TestArtifact", name=config["runtime_name"]) as run:
        artifact = wandb.Artifact(name=config["dataset_name"], type="dataset")
        artifact.add_dir(config["data_root"])
        run.log_artifact(artifact)
        run.finish()

def model_version_control(config):
    with wandb.init(entity="jjerry", job_type="mvc", project="TestArtifact", name=config["runtime_name"]) as run:
        artifact = wandb.Artifact(name=config["model_name"], type="model")
        run.use_artifact(f'{config["dataset_name"]}:latest')
        artifact.add_dir(config["ckpt_path"])
        run.log_artifact(artifact)
        run.finish()

job_type = "SetProject"

config = {
    "runtime_name":  f"Test{int(random.random()*10000)}",
    "data_root": "./data/cifar10",
    "dataset_name": f"CLASS{len(os.listdir(os.path.join('./data/cifar10', 'train')))}",
    "model_name": "efficientnet_b0",
}
config.update({"ckpt_path": f"./result/{config['runtime_name']}"})

start = time.time()
data_version_control(config)
print(f"DVC Elapsed Time: {time.time() - start}")


net = Model(config["model_name"])

save_checkpoint({'epoch': 1,
                'state_dict': net.state_dict()},
                is_best=False,
                checkpoint=config["ckpt_path"])

start = time.time()
model_version_control(config)
print(f"MVC Elapsed Time: {time.time() - start}")