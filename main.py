import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

import util
from dataset import pokemonDataset, ToDoubleTensor, Normalize, get_channel_mean_std
import dcgan as dc
import illustration_gan as illust
import config

PATH_IMAGE = '../pokemon_dataset/image'
PATH_TAG = '../pokemon_dataset/tags'
ARTWORK_TYPE = os.listdir(PATH_IMAGE)
PATH_MODEL = '../model'
if not os.path.exists(PATH_MODEL):
    os.makedirs(PATH_MODEL)

IS_ADD_I2V_TAG = False

def main():

    dataset = pokemonDataset(PATH_IMAGE, PATH_TAG, ['ken sugimori'], IS_ADD_I2V_TAG)

    # mean, std = get_channel_mean_std(dataset, DIM_IMG)
    # mean = [220.43362509, 217.50907014, 212.78514176]
    # std = [71.7985852,  73.64374336, 78.23258064]

    transform=transforms.Compose([ToDoubleTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    dataset.set_transform(transform)

    # conf = config.config_dcgan
    # net_gen, net_dis = dc.build_gen_dis(conf)
    # _, _, losses, imgs = dc.train(dataset, net_gen, net_dis, config.conf)

    conf = config.config_illustration_gan
    net_gen, net_dis = illust.build_gen_dis(conf)
    print(net_gen)
    print(net_dis)
    net_gen, net_dis, losses, imgs = illust.train(dataset, net_gen, net_dis, conf)

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




