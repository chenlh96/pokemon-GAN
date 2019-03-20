from importlib import reload
import dataset
import dcgan
import Func
from dataset import pokemonDataset, ToTensor
from dcgan import generator, discriminator, train
from os import listdir
import matplotlib.pyplot as plt
# import numpy as np

reload(dataset)
reload(dcgan)
reload(Func)


PATH_IMAGE = '../pokemon_dataset/image'
PATH_TAG = '../pokemon_dataset/tags'
ARTWORK_TYPE = listdir(PATH_IMAGE)
IS_ADD_I2V_TAG = False

test = pokemonDataset(PATH_IMAGE, PATH_TAG, ARTWORK_TYPE, IS_ADD_I2V_TAG, transform=ToTensor())


def main():
    pass