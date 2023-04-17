import json
import os
import random
import warnings
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN, ImageFolder

warnings.filterwarnings('ignore')

def build_cifar(use_cifar10=True, download=True):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),transforms.ToTensor()]

    if use_cifar10:
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = CIFAR10(root='~/datasets/',
                                train=True, download=download, transform=transform_train)
        val_dataset = CIFAR10(root='~/datasets/',
                              train=False, download=download, transform=transform_test)
        norm = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    else:
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = CIFAR100(root='~/datasets/',
                                 train=True, download=download, transform=transform_train)
        val_dataset = CIFAR100(root='~/datasets/',
                               train=False, download=download, transform=transform_test)
        norm = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    return train_dataset, val_dataset, norm

class CatsDogs(Dataset):
    def __init__(self, train=False, transform=None):
        if train:
            with open("/home/butong/datasets/dogs-vs-cats/train_list.json", "r") as f:
                self.file_list = json.load(f)
            train_transforms =  transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            self.transform = train_transforms
        else:
            with open("/home/butong/datasets/dogs-vs-cats/val_list.json", "r") as f:
                self.file_list = json.load(f)
            val_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            self.transform = val_transforms

    #dataset length
    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength
    
    #load an one of images
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)
        # note the difference on windows or linux
        label = img_path.split('/')[-1].split('.')[0]
        if label == 'dog':
            label = 1
        elif label == 'cat':
            label = 0
        return img_transformed, label