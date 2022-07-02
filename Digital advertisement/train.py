import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_mnist
from trainer import Trainer
from argparse import Trainer

config = {
    'train_ratio': .8,
    'batch_size' : 256,
    'n_epochs': 20,
    'verbose':1,
    'bt1_size': 2
}

config = Namespace(**config)

def show_image(x):
    if x.dim() == 1:
        x = x.view(int(x.size(0) **.5), -1)

    plt.imshow(x, cmap='gray')
    plt.show()

train_x,train_y = load_mnist(flatten=True)
test_x, test_y = load_mnist(is_train=False, flatten=True)

train_cnt = int(train_x.size(0) * config.train_ratio)
valid_cnt =train_x.size(0) - train_cnt

#Shuffle dataset to split into train/valid set
indices = torch.randperm(train_x.size(0))
train_x, valid_x = torch.index_select(
    train_x,
    dim= 0,
    index = indices
).split([train_cnt, valid_cnt], dim=0)


from model import Autoencoder
model = Autoencoder(btl_size=config.btl_size)
optimizer = optim.Adam(model.parameters())
crit = nn.MSELoss()

trainer = Trainer(model, optimizer, crit)

trainer.train((train_x,train_x), (valid_x, valid_x), config)