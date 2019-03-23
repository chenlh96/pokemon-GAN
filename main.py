from os import listdir
import matplotlib.pyplot as plt
import os
# import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

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
PATH_MODEL = '../model'
if not os.path.exists(PATH_MODEL):
    os.makedirs(PATH_MODEL)
IS_ADD_I2V_TAG = False

BATCH_SIZE = 10
DIM_IMG = 128
DIM_NOISE = 100

LEARNING_RATE = 0.0002
MOMENTUM = 0.5
EPOCHS = 1

def main():
    transform=transforms.Compose([
                        ToDoubleTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                   ])
    dataset = pokemonDataset(PATH_IMAGE, PATH_TAG, ARTWORK_TYPE, IS_ADD_I2V_TAG, transform=transform)

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

    net_gen, net_dis, losses, scores, imgs = train_base(EPOCHS, BATCH_SIZE, DIM_NOISE, DIM_IMG, device,
                                                        dataset, net_gen, net_dis, loss, optim_gen, optim_dis, PATH_MODEL, None)




if __name__ == "__main__":
    main()




