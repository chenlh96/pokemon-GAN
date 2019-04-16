import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from os import listdir
import os
import csv
from PIL import Image
import numpy as np

def get_channel_mean_std(dataset, dim_img, channel=3):
    mean = np.empty(channel)
    std = np.empty(channel)
    for i in range(len(dataset)):
        img = dataset[i][0]
        if type(img) == torch.Tensor:
            img = np.transpose(img.numpy(), (1, 2, 0))
        else:
            img = np.asarray(img)
        for j in range(channel):
            mean[j] += np.sum(img[:,:, j])
    mean /= dim_img ** 2 * len(dataset)
    
    for i in range(len(dataset)):
        img = dataset[i][0]
        if type(img) == torch.Tensor:
            img = np.transpose(img.numpy(), (1,2,0))
        for j in range(channel):
            std[j] += np.sum((img[:,:, j] - mean[j])** 2)
    std /= dim_img ** 2 * len(dataset)
    std = np.sqrt(std)
    return mean, std   

class pokemonDataset(Dataset):
    def __init__(self, image_dir, tag_dir, artwork_types=None, augmentation_types=None, is_add_i2v_tag=False, transform=None):
        self.image_dir = image_dir
        self.tag_dir = tag_dir
        self.artwork_types = None

        if artwork_types == None:
            self.artwork_types == listdir(self.image_dir)
        else:
            self.artwork_types = artwork_types
        
        self.augmentation_types = None
        if augmentation_types != None:
            self.augmentation_types == augmentation_types
        
        self.is_add_i2v_tag = is_add_i2v_tag
        self.sample_dir = None
        self.transform = transform

        self.load_sample_dir()

    def __len__(self):
        return len(self.sample_dir)

    def __getitem__(self, idx):
        sel_sample = self.sample_dir[idx]
        img = Image.open(sel_sample[0], 'r')
        
        if self.transform:
            img = self.transform(img)
        sel_sample = (img, sel_sample[1:])
        return sel_sample
    
    def set_transform(self, transform):
        if self.transform == None:
            self.transform = transform
        else:
            # todo: to check if the compose can accept a compose object as arg
            self.transform = transforms.Compose([self.transform, transform])

    def load_sample_dir(self):
        list_csv = listdir(self.tag_dir)
        csv_dict = {}
        for aw in self.artwork_types:
            csv_dict[aw] = None
            for csv_f in list_csv:
                if os.path.isfile(self.tag_dir + '/' + csv_f) and aw in csv_f:
                    if 'i2v' not in csv_f:
                        csv_dict[aw] = [self.tag_dir + '/' + csv_f]
                        print(self.is_add_i2v_tag)
                    if self.is_add_i2v_tag and 'i2v' in csv_f:
                        csv_dict[aw] = csv_dict[aw].append(self.tag_dir + '/' + csv_f)
                        print(csv_dict[aw])

        self.sample_dir = []
        for aw in self.artwork_types:
            list_tag = []
            with open(csv_dict[aw][0], 'r') as f:
                tagReader = csv.reader(f)
                next(tagReader, None)
                for row in tagReader:
                    list_tag.append(row[3:])
            if self.is_add_i2v_tag:
                with open(csv_dict[aw][1], 'r') as f:
                    tagReader = csv.reader(f)
                    next(tagReader, None)
                    for i, row in zip(tagReader, len(list_tag)):
                        list_tag[i] = list_tag[i] + row[1:]

            path_aug = self.image_dir + '/' + aw
            list_aug = listdir(path_aug)
            if self.augmentation_types != list_aug and self.augmentation_types != None:
                list_aug = self.augmentation_types
            
            for aug in list_aug:
                path_img = path_aug + '/' + aug
                list_img = listdir(path_img)

                for img, tag in zip(list_img, list_tag):
                    path_img_spec = path_img + '/' + img
                    self.sample_dir.append([path_img_spec] + tag)

class animeFaceDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        folder_list = listdir(self.image_dir)
        self.img_dir = []
        self.transform = transform

        for folder in folder_list:
            folder_dir = self.image_dir + '/' + folder
            if not os.path.isdir(folder_dir):
                continue
            
            file_list = listdir(folder_dir)
            if len(file_list) != 0:
                for file in file_list:
                    if file[0] != '.':
                        file_dir = folder_dir + '/' + file
                        self.img_dir.append(file_dir)

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        sel_sample = self.img_dir[idx]
        img = Image.open(sel_sample, 'r')
        
        if self.transform != None:
            img = self.transform(img)
        
        return [img]

class fmnist(Dataset):

    def __init__(self, root_dir, download, image_size = 64):
        self.root = root_dir
        self.download = download
        self.transform = [transforms.Resize(image_size), transforms.ToTensor()]
        transform_composite=transforms.Compose(self.transform)

        self.f_mnist_train = datasets.FashionMNIST(self.root + '/train/', train=True, download=self.download, transform=transform_composite)
        self.f_mnist_test = datasets.FashionMNIST(self.root + '/test/', train=False, download=self.download, transform=transform_composite)

    def __len__(self):
        return len(self.f_mnist_train) + len(self.f_mnist_test)

    def __getitem__(self, idx):
        if idx < len(self.f_mnist_train):
            return self.f_mnist_train[idx]
        else:
            return self.f_mnist_test[idx-len(self.f_mnist_train)]

