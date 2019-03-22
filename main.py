from os import listdir
import matplotlib.pyplot as plt
# import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import dataset
import dcgan
import Func
from dataset import pokemonDataset, ToDoubleTensor
from dcgan import generator, discriminator, train_base, init_weight

# from importlib import reload
# reload(dataset)
# reload(dcgan)
# reload(Func)


PATH_IMAGE = '../pokemon_dataset/image'
PATH_TAG = '../pokemon_dataset/tags'
ARTWORK_TYPE = listdir(PATH_IMAGE)
IS_ADD_I2V_TAG = False

BATCH_SIZE = 32
DIM_IMG = 128
DIM_NOISE = 100

STD_WEIGHT = 0.2
LEARNING_RATE = 0.1
MOMENTUM = 0.5
EPOCHS = 1

def main():
    dataset = pokemonDataset(PATH_IMAGE, PATH_TAG, ARTWORK_TYPE, IS_ADD_I2V_TAG, transform=ToDoubleTensor())

    device = torch.device("cpu")

    net_gen = generator(DIM_NOISE, DIM_IMG).to(device)
    net_gen.apply(init_weight)
    print(net_gen)
    net_dis = discriminator(DIM_IMG).to(device)
    net_dis.apply(init_weight)
    print(net_dis)

    loss = nn.BCELoss()
    optim_gen = optim.Adam(net_gen.parameters(), lr=LEARNING_RATE, betas=(MOMENTUM, 0.99))
    optim_dis = optim.Adam(net_dis.parameters(), lr=LEARNING_RATE, betas=(MOMENTUM, 0.99))

    net_gen, net_dis, losses, scores, imgs = train_base(EPOCHS, BATCH_SIZE, DIM_NOISE, DIM_IMG, dataset, net_gen, net_dis, loss, optim_gen, optim_dis, None)



if __name__ == "__main__":
    main()




