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
                # max_idx = list(spec_tags.values()).index(max_val)
                # img_tags.append(list(spec_tags.keys())[max_idx])
                img_tags.append(max_val)
            else:
                val = list(spec_tags.values())[0]
                if (val > threshold):
                    # img_tags.append('True')
                    img_tags.append(1)
                else:
                    # img_tags.append('False')
                    img_tags.append(0)
        with open(csv_dir, 'a', newline='') as f:
            tagWriter = csv.writer(f)
            tagWriter.writerow(np.concatenate([[img_f[:-4]], img_tags]))

def order_tags(path_csv_src, path_img_folder, path_csv_des=None):
    if path_csv_des == None:
        path_csv_des = path_csv_src
    tag_dict = {}
    type_list, ability_list, color_list = [], [], []
    with open(path_csv_src, 'r') as f:
        tagReader = csv.reader(f)
        for row in tagReader:
            tag_dict[row[0]] = row
            if row[3] not in type_list:
                type_list.append(row[3])
            if row[4] not in type_list:
                type_list.append(row[4])
            if row[5] not in ability_list:
                ability_list.append(row[5])
            if row[6] not in ability_list:
                ability_list.append(row[6])
            if row[7] not in color_list:
                color_list.append(row[7])
    
    type_dict = {k: (i / len(type_list)) for i, k in enumerate(type_list[1:])}
    ability_dict = {k: (i / len(ability_list)) for i, k in enumerate(ability_list[1:])}
    color_dict = {k: (i / len(color_list)) for i, k in enumerate(color_list[1:])}

    list_img_name = listdir(path_img_folder)
    
    with open(path_csv_des, 'w', newline = '') as f:
        tagWriter = csv.writer(f)
        tagWriter.writerow(['filename', 'index', 'name', 'type1', 'type2', 'ability1', 'ability2', 'color'])
        for name in list_img_name:
            r = tag_dict[name[:-4]]
            r[3] = type_dict[r[3]]
            r[4] = type_dict[r[4]]
            r[5] = ability_dict[r[5]]
            r[6] = ability_dict[r[6]]
            r[7] = color_dict[r[7]]
            tagWriter.writerow(r)

# make new dir
def makeImageFolderDir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# this is the main part of this script, we will do augmentation and tag the image in
# the following work

PATH_ORI = 'C:/Users/Leheng Chen/Desktop/HKUST/pokemon-GAN/pokemon_dataset/images_newori'
ARTWORK = listdir(PATH_ORI)
PATH_DATASET = 'C:/Users/Leheng Chen/Desktop/HKUST/pokemon-GAN/pokemon_dataset/image'

ORI_TYPE = 'ori'
AUGMENTATION_TYPE = ['flip', 'left rotation', 'right roration', 'contrast', 'saturation']

DIMENSION = 128
DEGREE_OF_ROTATE = 15
CONTRAST = 3
SATURATION = 1.5

for aw in ARTWORK:
    path_ori = PATH_ORI + '/' + aw
    dest_aug_folder = makeImageFolderDir(PATH_DATASET + '/' + aw)
    dest_ori = makeImageFolderDir(dest_aug_folder + '/' + ORI_TYPE)
    rescale_n_2rgb(path_ori, dest_ori, DIMENSION)
    for aug in AUGMENTATION_TYPE:
        dest_aug = makeImageFolderDir(dest_aug_folder + '/' + aug)
        augamentation(dest_ori, dest_aug, aug, DIMENSION, DEGREE_OF_ROTATE, CONTRAST, SATURATION)


# Now we first process the tags

PATH_TAG = 'C:/Users/Leheng Chen/Desktop/HKUST/pokemon-GAN/pokemon_dataset/tags/'
TAGS = ['eyes', 'smile', 'chibi', 'mouth', 'tail']
PATH_JSON = 'illustration2vec/tag_list.json'
THRESHOLD = 0.1
# import the illustration2vec model
illust2vec = illustration2vec.i2v.make_i2v_with_chainer("illustration2vec/illust2vec_tag_ver200.caffemodel", "illustration2vec/tag_list.json")

i2v_tag_dict = get_tag_values(PATH_JSON, TAGS)
i2v_tag_dict['smile'] = np.delete(i2v_tag_dict['smile'], 2)
i2v_tag_dict['tail'] = np.delete(i2v_tag_dict['tail'], [0, 1, 3, 5])

# To better looking the tags, we first list the tags by the order of the pokemon index(shown in the folder of origin image)

for aw in ARTWORK:
    path_csv = PATH_TAG + 'pokemon_tag_' + aw + '.csv'
    path_csv_des = PATH_TAG + 'pokemon_tag_' + aw + '_code.csv'
    path_img_folder = PATH_ORI + '/' + aw
    order_tags(path_csv, path_img_folder, path_csv_des)


for aw in ARTWORK:
    csv_dir = PATH_TAG + 'pokemon_tag_' + aw + '_i2v.csv'
    img_floder = PATH_ORI + '/' + aw
    get_i2v_tags_2csv(csv_dir, img_floder, illust2vec, i2v_tag_dict, THRESHOLD)

