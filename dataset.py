import torch
from torch.utils.data import Dataset
from os import listdir
import os
import csv
from PIL import Image
import numpy as np

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, tags = sample['image'], sample['tags']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'tags': torch.from_numpy(tags)}

class pokemonDataset(Dataset):
    def __init__(self, image_dir, tag_dir, artwork_types=None, is_add_i2v_tag=False):
        self.image_dir = image_dir
        self.tag_dir = tag_dir
        self.artwork_types = None
        if artwork_types == None:
            self.artwork_types == listdir(self.image_dir)
        else:
            assert artwork_types == listdir(self.image_dir)
            self.artwork_types = artwork_types
        self.is_add_i2v_tag = is_add_i2v_tag
        self.sample_dir  = None

        self.load_sample_dir()

    def __len__(self):
        return len(self.sample_dir)

    def __getitem__(self, idx):
        sel_sample = self.sample_dir[idx]
        img = Image.open(sel_sample[0], 'r')
        img_2arr = np.asarray(img)
        return {'image': img_2arr, 'tags': sel_sample[1:]}


    def load_sample_dir(self):
        list_csv = listdir(self.tag_dir)
        csv_dict = {}
        for csv in list_csv:
            for aw in self.artwork_types:
                if os.path.isfile(csv) and aw in csv:
                    if 'i2v' not in csv:
                        csv_dict[aw] = [self.tag_dir + '/' + csv]
                    if self.is_add_i2v_tag and 'i2v' in csv:
                        csv_dict[aw] = csv_dict[aw].append(self.tag_dir + '/' + csv)

        self.sample_dir = []
        for aw in self.artwork_types:
            list_tag = []
            with open(csv_dict[aw][0], 'r') as f:
                tagReader = csv.reader(f)
                next(tagReader, None)
                for row in tagReader:
                    list_tag.append(row[4:])
            if self.is_add_i2v_tag:
                with open(csv_dict[aw][1], 'r') as f:
                    tagReader = csv.reader(f)
                    next(tagReader, None)
                    for i, row in zip(tagReader, len(list_tag)):
                        list_tag[i] = list_tag[i] + row[1:]

            path_aug = self.image_dir + '/' + aw
            list_aug = listdir(path_aug)
            for aug in list_aug:
                path_img = path_aug + '/' + aug
                list_img = listdir(path_img)

                for tag, img in zip(list_img, list_tag):
                    path_img_spec = path_img + '/' + img
                    self.sample_dir.append([path_img_spec] + tag)
            

    def add_i2v_tag(self):
        pass
