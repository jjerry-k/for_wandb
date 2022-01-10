import random
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class BaseDataset(Dataset):

    def __init__(self, data_dir, transform):
        
        self.filelist = []
        self.classes = sorted(os.listdir(data_dir))
        for root, sub_dir, files in os.walk(data_dir):
            if not len(files): continue
            files = [os.path.join(root, file) for file in files if file.endswith("jpg")]
            self.filelist += files
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filelist)

    def __getitem__(self, idx):

        image = Image.open(self.filelist[idx])
        image = self.transform(image)
        label = self.filelist[idx].split('/')[-2]
        label = self.classes.index(label)
        return image, label

def Dataloader(config, transform=None):

    input_size = (config["INPUTSIZE"], config["INPUTSIZE"])

    dataloaders = {}

    for split in ['train', 'validation']:
        path = os.path.join("./data/", config["NAME"], split)
        transform_list = [transforms.Resize(input_size), transforms.ToTensor()]
        if split == 'train':
            transform_list = transform_list.insert(1, transform) if transform else transform_list
            dl = DataLoader(BaseDataset(path, transforms.Compose(transform_list)), batch_size=config["BATCHSIZE"], shuffle=True, num_workers=config["NUMWORKER"], drop_last=True)
        else:
            dl = DataLoader(BaseDataset(path, transforms.Compose(transform_list)), batch_size=config["BATCHSIZE"], shuffle=False, num_workers=config["NUMWORKER"])

        dataloaders[split] = dl

    return dataloaders