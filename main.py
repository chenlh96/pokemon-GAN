from os import listdir
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt

import dataset
import dcgan
import Func
from dataset import pokemonDataset, ToDoubleTensor, Normalize
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

BATCH_SIZE = 5
DIM_IMG = 128
DIM_NOISE = 100

LEARNING_RATE = 0.0002
MOMENTUM = 0.5
EPOCHS = 1

def main():
    dataset = pokemonDataset(PATH_IMAGE, PATH_TAG, ARTWORK_TYPE, IS_ADD_I2V_TAG)
    # mean, std = dataset.get_channel_mean_std(DIM_IMG)
    mean = [220.43362509, 217.50907014, 212.78514176]
    std = [5155.03683635, 5423.40093651, 6120.33667355]
    transform=transforms.Compose([ToDoubleTensor(), Normalize(mean, std)])
    dataset.set_transform(transform)
    # sample = dataset[0][0].numpy()
    # img = np.transpose(sample, (1,2,0)).astype(int)
    # plt.figure()
    # plt.imshow(img)
    # plt.show()


    device = torch.device("cpu")

    net_gen = generator(DIM_NOISE, DIM_IMG).to(device)
    net_gen.apply(init_weight)
    # print(net_gen)
    net_dis = discriminator(DIM_IMG).to(device)
    net_dis.apply(init_weight)
    # print(net_dis)

    loss = nn.BCELoss()
    optim_gen = optim.Adam(net_gen.parameters(), lr=LEARNING_RATE, betas=(MOMENTUM, 0.99))
    optim_dis = optim.Adam(net_dis.parameters(), lr=LEARNING_RATE, betas=(MOMENTUM, 0.99))

    net_gen, net_dis, _, _, _ = train_base(EPOCHS, BATCH_SIZE, DIM_NOISE, DIM_IMG, device,
                                                        dataset, net_gen, net_dis, loss, optim_gen, optim_dis, PATH_MODEL)


if __name__ == "__main__":
    main()




