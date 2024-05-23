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
from PIL import ImageFile
# from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings('ignore')

# code from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py 
# Improved Regularization of Convolutional Neural Networks with Cutout.
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def cifar_dataset(use_cifar10=True, download=False):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),transforms.ToTensor()]

    if use_cifar10:
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = CIFAR10(root='E:\datasets',
                                train=True, download=download, transform=transform_train)
        val_dataset = CIFAR10(root='E:\datasets',
                              train=False, download=download, transform=transform_test)
        norm = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    else:
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = CIFAR100(root='E:\datasets',
                                 train=True, download=download, transform=transform_train)
        val_dataset = CIFAR100(root='E:\datasets',
                               train=False, download=download, transform=transform_test)
        norm = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    return train_dataset, val_dataset, norm

class Cutout_dvs(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h = img.size(2)
        w = img.size(3)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img
    
def function_nda(data, M=1, N=2):
    c = 15 * N
    rotate_tf = transforms.RandomRotation(degrees=c)
    e = 8 * N
    cutout_tf = Cutout_dvs(length=e)

    def roll(data, N=1):
        a = N * 2 + 1
        off1 = random.randint(-a, a)
        off2 = random.randint(-a, a)
        return torch.roll(data, shifts=(off1, off2), dims=(2, 3))

    def rotate(data, N):
        return rotate_tf(data)

    def cutout(data, N):
        return cutout_tf(data)

    transforms_list = [roll, rotate, cutout]
    sampled_ops = np.random.choice(transforms_list, M)
    for op in sampled_ops:
        data = op(data, N)
    return data


    
def build_dvscifar(root):
    def trans_t(data):
        resize = transforms.Resize(size=(48, 48))  # 48 48
        data = torch.from_numpy(data) 
        data = resize(data).float()
        flip = random.random() > 0.5
        if flip:
            data = torch.flip(data, dims=(3,))
        data = function_nda(data)
        return data
    
    def tt(data):
        aug = transforms.Resize(size=(48, 48))
        data = torch.from_numpy(data)
        data = aug(data).float()
        return data

    data1 = CIFAR10DVS(root=root, data_type='frame', frames_number=10, split_by='number', transform=trans_t)
    train_dataset, _ = torch.utils.data.random_split(data1, [9000, 1000], generator=torch.Generator().manual_seed(42))
    data2 = CIFAR10DVS(root=root, data_type='frame', frames_number=10, split_by='number', transform=tt)
    _, val_dataset = torch.utils.data.random_split(data2, [9000, 1000], generator=torch.Generator().manual_seed(42))
    return train_dataset, val_dataset, None