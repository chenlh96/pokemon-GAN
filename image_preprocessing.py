import numpy as np
import os
from os import listdir
from PIL import Image, ImageEnhance
import csv
import illustration2vec.i2v
import json


# fix the figure size to 256 * 256 or 64 * 64 and turn the figures into RGB mode, where we change
# its format to .jpg

def rescale_n_2rgb(ori_dir, dest_dir, dimension):
    list_img = listdir(ori_dir)
    for im in list_img:
        img_ori = Image.open(ori_dir + '/' + im, 'r').convert('RGBA')
        x, y = img_ori.size
        max_size = max(x, y)
        img_new = Image.new('RGB', (max_size, max_size), (255, 255, 255))
        img_new.paste(img_ori, (int((max_size - x) / 2), int((max_size - y) / 2)), mask=img_ori.split()[3])
        img_new = img_new.resize((dimension,dimension))
        img_new.save(dest_dir + '/' + im[:-3]+'jpg', 'JPEG', quality=100)
    

# Now I am going to start data augmentaiton step sinse the original figures set are too small
# to train a deep learning model

def augamentation(ori_dir, dest_dir, augmentation_type, dimension, degree_of_rotation, contrast, satuation):
    list_img = listdir(ori_dir)
    for im in list_img:
        img_ori = Image.open(ori_dir + '/' + im, 'r')
        if augmentation_type == 'flip':
            img_new = img_ori.transpose(Image.FLIP_LEFT_RIGHT)
        elif augmentation_type == 'left rotation':
            img_new = img_ori.rotate(360 - degree_of_rotation, resample=Image.NEAREST, expand=1, fillcolor='white')
            img_new = img_new.resize((dimension, dimension))
        elif augmentation_type == 'right roration':
            img_new = img_ori.rotate(degree_of_rotation, resample=Image.NEAREST, expand=1, fillcolor='white')
            img_new = img_new.resize((dimension, dimension))
        elif augmentation_type == 'contrast':
            img_new = ImageEnhance.Contrast(img_ori).enhance(contrast)
        elif augmentation_type == 'saturation':
            img_new = ImageEnhance.Brightness(img_ori).enhance(satuation)
        else:
            img_new = img_ori
        img_new.save(dest_dir + '/' + im[:-4] + '_' + augmentation_type + '.jpg', 'JPEG', quality=100)


# next we will use illustration 2 vec to identify some features of the look of a pokemon, including:
# eyes, smile, chibi(child look), mouth, tail

# get all the possible values of that tags

def get_tag_values(json_dir, tags):
    i2v_tag_dict = {}
    with open(json_dir) as f:
        data = json.load(f)
        print(len(data))
        for t in tags:
            vals = list()
            for d in data:
                if t in d:
                    vals.append(d)
            i2v_tag_dict[t] = np.asarray(vals)
    return i2v_tag_dict

# wirte down the value to the csv

def get_i2v_tags_2csv(csv_dir, img_folder_dir, illust2vec, i2v_tag_dict, threshold):
    with open(csv_dir, 'w', newline='') as f:
        tagWriter = csv.writer(f)
        tagWriter.writerow(np.concatenate([['filename'], TAGS]))

    # start to tag all the images
    list_img = listdir(img_folder_dir)
    for img_f in list_img:
        img = Image.open(img_folder_dir + '/' + img_f)
        img_tags = list()
        for t in TAGS:
            spec_tags = illust2vec.estimate_specific_tags([img], i2v_tag_dict[t])[0]
            if (len(i2v_tag_dict[t]) > 1):
                max_val = max(spec_tags.values())
                max_idx = list(spec_tags.values()).index(max_val)
                img_tags.append(list(spec_tags.keys())[max_idx])
            else:
                val = list(spec_tags.values())[0]
                if (val > threshold):
                    img_tags.append('True')
                else:
                    img_tags.append('False')
        with open(csv_dir, 'a', newline='') as f:
            tagWriter = csv.writer(f)
            tagWriter.writerow(np.concatenate([[img_f[:-4]], img_tags]))


# this is the main part of this script, we will do augmentation and tag the image in
# the following work

PATH_ORI = 'C:/Users/Leheng Chen/Desktop/HKUST/pokemon-GAN/pokemon_dataset/images_newori'
ARTWORK = listdir(PATH_ORI)
PATH_DATASET = 'C:/Users/Leheng Chen/Desktop/HKUST/pokemon-GAN/pokemon_dataset/image'

ORI_TYPE = 'ori'
AUGMENTATION_TYPE = ['flip', 'left rotation', 'right roration', 'contrast', 'saturation']

if not os.path.exists(PATH_DATASET):
    os.makedirs(PATH_DATASET)
for aw in ARTWORK:
    path_folder = PATH_DATASET+'/'+aw
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
    for aug in np.concatenate([[ORI_TYPE], AUGMENTATION_TYPE]):
        path_folder = PATH_DATASET+'/'+aw+'/'+aug
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)

DIMENSION = 256
DEGREE_OF_ROTATE = 30
CONTRAST = 5
SATURATION = 2

for aw in ARTWORK:
    path_ori = PATH_ORI + '/' + aw
    dest_ori = PATH_DATASET + '/' + aw + '/' + ORI_TYPE
    rescale_n_2rgb(path_ori, dest_ori, DIMENSION)
    dest_aug_folder = PATH_DATASET + '/' + aw
    for aug in AUGMENTATION_TYPE:
        dest_aug = dest_aug_folder + '/' + aug
        augamentation(dest_ori, dest_aug, aug, DIMENSION, DEGREE_OF_ROTATE, CONTRAST, SATURATION)

# Now we first process the tags
# To better looking the tags, we first list the tags by the order of the pokemon index(shown in the folder of origin image)

PATH_TAG = '../pokemon_dataset/tags/'

for aw in ARTWORK:
    tag_dict = {}
    with open(PATH_TAG + 'pokemon_tag_' + aw + '.csv', 'r') as f:
        tagReader = csv.reader(f)
        for row in tagReader:
            tag_dict[row[0]] = row

    list_img_name = listdir(PATH_ORI + '/' + aw)
    
    with open(PATH_TAG + 'pokemon_tag_' + aw + '.csv', 'w', newline = '') as f:
        tagWriter = csv.writer(f)
        tagWriter.writerow(['filename', 'index', 'name', 'type1', 'type2', 'ability1', 'ability2', 'color'])
        for name in list_img_name:
            tagWriter.writerow(tag_dict[name[:-4]])


TAGS = ['eyes', 'smile', 'chibi', 'mouth', 'tail']
PATH_JSON = 'illustration2vec/tag_list.json'
THRESHOLD = 0.1
# import the illustration2vec model
illust2vec = illustration2vec.i2v.make_i2v_with_chainer("illustration2vec/illust2vec_tag_ver200.caffemodel", "illustration2vec/tag_list.json")

i2v_tag_dict = get_tag_values(PATH_JSON, TAGS)
i2v_tag_dict['smile'] = np.delete(i2v_tag_dict['smile'], 2)
i2v_tag_dict['tail'] = np.delete(i2v_tag_dict['tail'], [0, 1, 3, 5])

for aw in ARTWORK[2:3]:
    csv_dir = PATH_TAG + 'pokemon_tag_' + aw + '_i2v.csv'
    img_floder = PATH_ORI + '/' + aw
    get_i2v_tags_2csv(csv_dir, img_floder, illust2vec, i2v_tag_dict, THRESHOLD)

