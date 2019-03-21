from os import listdir
import matplotlib.pyplot as plt
# import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import dataset
import dcgan
import Func
from dataset import pokemonDataset, ToTensor
from dcgan import generator, discriminator, train_base

from importlib import reload
reload(dataset)
reload(dcgan)
reload(Func)


PATH_IMAGE = '../pokemon_dataset/image'
PATH_TAG = '../pokemon_dataset/tags'
ARTWORK_TYPE = listdir(PATH_IMAGE)
IS_ADD_I2V_TAG = False

BATCH_SIZE = 126
DIM_IMG = 128
DIM_NOISE = 100

STD_NOISE = 0.2
LEARNING_RATE = 0.0002
MOMENTUM = 0.5
EPOCHS = 1000

def main():
    dataset = pokemonDataset(PATH_IMAGE, PATH_TAG, ARTWORK_TYPE, IS_ADD_I2V_TAG, transform=ToTensor())

    net_gen = generator(DIM_NOISE, DIM_IMG)
    net_dis = discriminator(DIM_IMG)

    loss = nn.BCELoss()
    optim_gen = optim.Adam(net_gen.parameters, LEARNING_RATE, MOMENTUM)
    optim_dis = optim.Adam(net_dis.parameters, LEARNING_RATE, MOMENTUM)

    train_base(EPOCHS, BATCH_SIZE, DIM_NOISE, dataset, net_gen, net_dis, loss, optim_gen, optim_dis)

if name == "__main__":
    main()




