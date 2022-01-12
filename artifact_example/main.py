# import os
# import shutil

# import time
# import datetime

# import wandb
# import random
# import torch

# import model
# import utils



# job_type = "SetProject"

# config = {
#     "runtime_name":  f'{datetime.datetime.now().strftime("%Y_%m_%d/%H_%M_%S")}',
#     "data_root": "./data/cifar10",
#     "dataset_name": f"CLASS{len(os.listdir(os.path.join('./data/cifar10', 'train')))}",
#     "model_name": "efficientnet_b1",
# }
# config.update({"ckpt_path": f"./result/{config['runtime_name']}"})

# # start = time.time()
# # data_version_control(config)
# # print(f"DVC Elapsed Time: {time.time() - start}")


# net = model.Model(config["model_name"])

# utils.save_checkpoint({'epoch': 1,
#                 'state_dict': net.state_dict()},
#                 is_best=False,
#                 checkpoint=config["ckpt_path"])

# start = time.time()
# model_version_control(config, 'train')
# print(f"MVC Elapsed Time: {time.time() - start}")

# start = time.time()
# model_version_control(config, 'test')
# print(f"MVC Elapsed Time: {time.time() - start}")


import os
import yaml
import shutil
import logging
import datetime
import argparse

import numpy as np
from pprint import pprint

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms

import model
import trainer
import utils

def data_version_control(config):
    for mode in ["train", "test"]:
        with wandb.init(entity="jjerry", job_type="dvc", project="TestArtifact") as run:
            print(f"{mode.capitalize()} Dataset Version Control")
            artifact = wandb.Artifact(name=f'{mode}_dataset', type="dataset", description=config["DATA"]["NAME"])
            artifact.add_dir(os.path.join(config["DATA"]["ROOT"], config["DATA"]["NAME"], mode))
            run.log_artifact(artifact)
            run.finish()

def model_version_control(config, mode):
    with wandb.init(entity="jjerry", job_type="mvc", project="TestArtifact") as run:
        artifact = wandb.Artifact(name=f'{config["MODEL"]["BASEMODEL"]}', type="model")
        run.use_artifact(f'{mode}_dataset:latest')
        artifact.add_file(os.path.join(config["COMMON"]["SAVEPATH"], "best.pth.tar"))

        run.log_artifact(artifact)
        run.finish()

optm_dict = {
    "adadelta": optim.Adadelta,
    "adam": optim.Adam,
    "asgd": optim.ASGD,
    "adadelta": optim.Adadelta,
    "adagrad": optim.Adagrad,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "adamax": optim.Adamax,
    "rmsprop": optim.RMSprop,
    "rprop": optim.Rprop,
    "sgd": optim.SGD,
    "sparseadam": optim.SparseAdam
}

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.yaml',
                    help="Path of configuration file")    

def main(config):
    # SET DEVICE
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= ",".join(str(gpu) for gpu in config["COMMON"]["GPUS"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    DATE = datetime.datetime.now().strftime("%Y_%m_%d/%H_%M_%S")
    
    SAVEPATH = os.path.join(config["COMMON"]["SAVEPATH"], DATE)
    config["COMMON"]["SAVEPATH"] = SAVEPATH
    os.makedirs(SAVEPATH)
    utils.set_logger(os.path.join(SAVEPATH, "train.log"))
    utils.write_yaml(os.path.join(SAVEPATH, "config.yaml"), config)

    # DATA LOADING
    logging.info(f'Loading {config["DATA"]["NAME"]} datasets')
    transform = [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)]
    loader = trainer.Dataloader(config["DATA"])
    # check configuration & real
    num_classes = len(loader["train"].dataset.classes)
    assert num_classes == config["MODEL"]["NUMCLASSES"], f'Number of class is not same!\nIn Directory: {num_classes}\nIn Configuration: {config["MODEL"]["NUMCLASSES"]}'

    # PREPROCESSING
    # Add New class
    # Add New data 

    # MODEL BUILD
    logging.info(f"Building model")
    net = model.Model(config["MODEL"]["BASEMODEL"], config["MODEL"]["NUMCLASSES"], config["MODEL"]["FREEZE"]).to(device)
    # net = model.Model(num_classes=config["MODEL"]["NUMCLASSES"]).to(device)
    
    if torch.cuda.is_available() and len(config["COMMON"]["GPUS"]) > 1:
        logging.info(f"Multi GPU mode")
        net = torch.nn.DataParallel(net, device_ids=config["COMMON"]["GPUS"]).to(device)

    criterion = model.loss_fn
    metrics = {"acc": model.accuracy} # If classification
    # metrics = {}
    optm = optm_dict[config["TRAIN"]["OPTIMIZER"]](net.parameters(), lr=config["TRAIN"]["LEARNINGRATE"])
    
    # TRAINING
    EPOCHS = config["TRAIN"]["EPOCHS"]
    logging.info(f"Training start !")
    best_val_loss = np.inf
    for epoch in range(EPOCHS):

        metrics_summary = trainer.train(epoch, net, optm, criterion, loader["train"], metrics, device, config)
        metrics_summary.update(trainer.eval(epoch, net, optm, criterion, loader["validation"], metrics, device, config))

        metrics_string = " ; ".join(f"{key}: {value:05.3f}" for key, value in metrics_summary.items())
        logging.info(f"[{epoch+1}/{EPOCHS}] Performance: {metrics_string}")

        is_best = metrics_summary['val_loss'] <= best_val_loss

        utils.save_checkpoint({'epoch': epoch + 1,
                                'state_dict': net.state_dict(),
                                'optim_dict': optm.state_dict()},
                                is_best=is_best,
                                checkpoint=SAVEPATH)
        
        if is_best:
            logging.info("Found new best loss !")
            best_val_loss = metrics_summary['val_loss']

            best_json_path = os.path.join(
                SAVEPATH, "metrics_best.json")
            utils.save_dict_to_json(metrics_summary, best_json_path, is_best)

        last_json_path = os.path.join(
            SAVEPATH, "metrics_history.json")
        utils.save_dict_to_json(metrics_summary, last_json_path)

    # Data version control

    logging.info(f"Training done !")


if __name__ == "__main__":
    
    # Load config file
    args = parser.parse_args()
    config = utils.config_parser(args.config)

    # Execute main function
    main(config)