import torch
import torch.nn as nn
import torchvision
from torch.utils import data
from torchvision import datasets, transforms
from d2l import torch as d2l
d2l.use_svg_display()

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=trans, download=True)

len(mnist_train), len(mnist_test)
print(mnist_train[0][0].shape)

batch_size = 128

def get_dataloader_workers():
    return 4

train_inter = data.DataLoader(mnist_train, batch_size=batch_size, num_workers=get_dataloader_workers(),shuffle=True)

timer = d2l.Timer()

for X,y in train_inter:
    continue

