from importlib import reload
import dataset
reload(dataset)
from dataset import pokemonDataset, ToTensor
from torch.utils.data import DataLoader
from torchvision import transforms
from os import listdir
import matplotlib.pyplot as plt
import numpy as np


PATH_IMAGE = '../pokemon_dataset/image'
PATH_TAG = '../pokemon_dataset/tags'
ARTWORK_TYPE = listdir(PATH_IMAGE)
IS_ADD_I2V_TAG = False

test = pokemonDataset(PATH_IMAGE, PATH_TAG, ARTWORK_TYPE, IS_ADD_I2V_TAG, transform=transforms.Compose([ToTensor()]))

test_loader = DataLoader(test, batch_size=64, shuffle=True)