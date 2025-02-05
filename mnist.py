#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ]
)

batch_size = 8

trainval_set = torchvision.datasets.MNIST(root='./data',
                                          train=True,
                                          transform=transform,
                                          download=True)

trainval_loader = torch.utils.data.DataLoader(trainval_set,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=2)

test_set = torchvision.datasets.MNIST(root='./data',
                                      train=False,
                                      download=True,
                                      transform=transform)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=2)


def img_show(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_pixel_values(image):
    f, ax = plt.subplots(figsize=(16, 16))
    sns.heatmap(image, annot=True, fmt='.1f', square=True, cmap="YlGnBu", ax=ax)
    plt.show(f)


def show_batch_images(loader):
    pass

#skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#def stratifiedKFold():
#    for train_index, val_index, in 


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(1, 4, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4,  128)
        self.fc1 = nn.Linear(128, 32)
        self.fc1 = nn.Linear(F.relu(self, conv2d[x]))

    

def main():
    show_pixel_values(trainval_set[0])
    pass


if __name__=='__main__':
    main()
