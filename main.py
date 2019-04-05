from os import listdir
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import util
from dataset import pokemonDataset, ToDoubleTensor, Normalize, get_channel_mean_std
from dcgan import generator, discriminator, train_base, init_weight

PATH_IMAGE = '../pokemon_dataset/image'
PATH_TAG = '../pokemon_dataset/tags'
ARTWORK_TYPE = listdir(PATH_IMAGE)
PATH_MODEL = '../model'
if not os.path.exists(PATH_MODEL):
    os.makedirs(PATH_MODEL)
PATH_MODEL = PATH_MODEL + '/dcgan.pth'
IS_ADD_I2V_TAG = False

BATCH_SIZE = 16
DIM_IMG = 64
DIM_NOISE = 100
LEARNING_RATE = 0.0002
MOMENTUM = 0.5
EPOCHS = 1
INIT = True
DEVICE = torch.device("cpu")

def main():

    dataset = pokemonDataset(PATH_IMAGE, PATH_TAG, ['ken sugimori'], IS_ADD_I2V_TAG)

    # mean, std = get_channel_mean_std(dataset, DIM_IMG)
    # mean = [220.43362509, 217.50907014, 212.78514176]
    # std = [71.7985852,  73.64374336, 78.23258064]

    transform=transforms.Compose([ToDoubleTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    dataset.set_transform(transform)

    net_gen = generator(DIM_NOISE, DIM_IMG).to(DEVICE)
    net_dis = discriminator(DIM_IMG).to(DEVICE)
    print(net_gen)
    print(net_dis)

    if INIT:
        net_gen.apply(init_weight)
        net_dis.apply(init_weight)
    else:
        ext = PATH_MODEL[-4]
        path_model = PATH_MODEL[:-4] + '_epoch_%d' + ext % EPOCHS
        net_gen, net_dis = util.load_checkpoint(EPOCHS, net_gen, net_dis, path_model)
    
    loss = nn.BCELoss()
    optim_gen = optim.Adam(net_gen.parameters(), lr=LEARNING_RATE, betas=(MOMENTUM, 0.99))
    optim_dis = optim.Adam(net_dis.parameters(), lr=LEARNING_RATE, betas=(MOMENTUM, 0.99))

    net_gen, net_dis, losses, _, imgs = train_base(EPOCHS, BATCH_SIZE, DIM_NOISE, DEVICE,
                                                    dataset, net_gen, net_dis, loss, optim_gen, optim_dis, PATH_MODEL)

    plt.figure(figsize=(20, 10))
    plt.plot(losses[0], label = 'generator')
    plt.plot(losses[1], label = 'discriminator')
    plt.title('Loss of training the gennerator and discriminator')
    plt.xlabel('loss')
    plt.ylabel('process')
    plt.legend()
    plt.show()

    grid_img = util.make_figure_grid(imgs[0], 8)
    plt.figure()
    plt.imshow(grid_img)
    plt.show()


if __name__ == "__main__":
    main()




