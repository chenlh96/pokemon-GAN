import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets

import dcgan as dc
import illustration_gan as illust
import hr_anime_gan as hranime
import config
import utils.util as util
import dataset as dset

PATH_IMAGE = '../pokemon_dataset/image'
PATH_TAG = '../pokemon_dataset/tags'
ARTWORK_TYPE = os.listdir(PATH_IMAGE)
PATH_MODEL = '../model'
if not os.path.exists(PATH_MODEL):
    os.makedirs(PATH_MODEL)

IS_ADD_I2V_TAG = False

def main():
    # ------------------- create the dataset -----------------------------
    f_minst = dset.fmnist('../', download=False, image_size=64)

    transform=transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
    anime = dset.animeFaceDataset('../anime_face_dataset/anime-faces', transform=transform)

    dataset = dset.pokemonDataset(PATH_IMAGE, PATH_TAG, ['ken sugimori'], IS_ADD_I2V_TAG)
    # mean, std = dset.get_channel_mean_std(dataset, DIM_IMG)
    # mean = [220.43362509, 217.50907014, 212.78514176]
    # std = [71.7985852,  73.64374336, 78.23258064]
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    dataset.set_transform(transform)

    for i in range(20):
        util.imshow(anime[i][0])
    return


    # ------------------- build and train the model -----------------------------
    # 1. DCGAN
    # CONFIG = config.config_dcgan
    # net_gen, net_dis = dc.build_gen_dis(CONFIG)
    # _, _, losses, imgs = dc.train(dataset, net_gen, net_dis, CONFIG)

    # 2. illustration GAN
    CONFIG = config.config_illustration_gan
    net_gen, net_dis = illust.build_gen_dis(CONFIG)
    print(net_gen)
    print(net_dis)
    
    net_gen, net_dis, losses, imgs = illust.train(anime, net_gen, net_dis, CONFIG)

    # 3. anime GAN in the 2017 paper
    CONFIG = config.config_illustration_gan
    net_gen, net_dis = illust.build_gen_dis(CONFIG)
    print(net_gen)
    print(net_dis)
    
    net_gen, net_dis, losses, imgs = illust.train(anime, net_gen, net_dis, CONFIG)


    # ------------------ visualize the result and evaluation --------------------------------
    plt.figure(figsize=(20, 10))
    plt.plot(losses[0], label = 'generator')
    plt.plot(losses[1], label = 'discriminator')
    plt.title('Loss of training the gennerator and discriminator')
    plt.xlabel('loss')
    plt.ylabel('process')
    plt.legend()
    plt.show()

    plt.figure()
    for i in range(CONFIG.EPOCHS):
        grid_img = util.make_figure_grid(imgs[i], 8)
        plt.imshow(grid_img)
    plt.show()


if __name__ == "__main__":
    main()

