# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

from pokeball.items import PokemonData, PokemonPhoto, PokemonPhoto2GetTag
import urllib
import os
import csv
from PIL import Image
import numpy as np
from os import listdir

class PokeballPipeline(object):
    def __init__(self, IMAGE_STORE, FILE_STORE):
        # if IMAGE_STORE is None or MAXIMUM_IMAGE_NUMBER is None:
        #     raise CloseSpider('Pipeline load settings failed')
        self.IMAGES_STORE = IMAGE_STORE
        self.FILES_STORE = FILE_STORE
        self.artwork_types = ['ken sugimori', 'dream work', 'os anime']

    @classmethod
    def from_crawler(cls, crawler):
        settings = crawler.settings
        return cls(settings['IMAGES_STORE'], settings['FILES_STORE'])

    def open_spider(self, spider):
        # make image file folder:
        if spider.name == "pokeball":
            # open/create a csv file to store tag
            if not os.path.exists(self.FILES_STORE):
                os.makedirs(self.FILES_STORE)
            if not os.path.exists(self.IMAGES_STORE):
                os.makedirs(self.IMAGES_STORE)
            
            self.csv_file_name = "{}/{}".format(self.FILES_STORE, 'pokemon_tag.csv')
            with open(self.csv_file_name, 'w', newline='') as f:
                self.tagWriter = csv.writer(f)
                self.tagWriter.writerow(['name', 'type1', 'type2', 'ability1', 'ability2', 'color'])
        
        if spider.name == 'masterball':
            csv_file_name_read = "{}/{}".format(self.FILES_STORE, 'pokemon_tag.csv')
            with open(csv_file_name_read, 'r') as f:
                tagreader = csv.reader(f)

                self.tag_dict = {}
                for row in tagreader:
                    # print(row)
                    self.tag_dict[row[1]] = row

            for aw in self.artwork_types:
                self.csv_file_name_total = "{}/{}".format(self.FILES_STORE, 'pokemon_tag_'+ aw + '.csv')
                with open(self.csv_file_name_total, 'w', newline='') as f:
                    self.tagWriter = csv.writer(f)
                    self.tagWriter.writerow(['filename', 'index', 'name', 'type1', 'type2', 'ability1', 'ability2', 'color'])

            self.IMAGES_STORE_2 = '../pokemon_dataset/images_newori'
            if not os.path.exists(self.IMAGES_STORE_2):
                os.makedirs(self.IMAGES_STORE_2)

            for aw in self.artwork_types:
                image_path = self.IMAGES_STORE_2 + '/' + aw
                if not os.path.exists(image_path):
                    os.makedirs(image_path)               

    
    def close_spider(self, spider):
        for aw in self.artwork_types:
            self.csv_file_name_total = "{}/{}".format(self.FILES_STORE, 'pokemon_tag_' + aw + '.csv')
            with open(self.csv_file_name_total, 'r') as f:
                tagreader = csv.reader(f)
                tag_dict_mess = {}
                for row in tagreader:
                    # print(row)
                    tag_dict_mess[row[0]] = row
            
            image_path = self.IMAGES_STORE_2 + '/' + aw
            list_img_name = listdir(image_path)

            with open(self.csv_file_name_total, 'w', newline='') as f:
                tagRewriter = csv.writer(f)
                tagRewriter.writerow(['filename', 'index', 'name', 'type1', 'type2', 'ability1', 'ability2', 'color'])
                for img_name in list_img_name:
                    img = img_name.split('.')[0]
                    tagRewriter.writerow(tag_dict_mess[img])

    def process_photo_2tag(self, item):
        name = item['pokemon_name'].replace('.', '')
        size = item['size']
        folder = item['artwork']
        if size > 10 and 'Ash' not in name and 'card' not in name and 'Sugimori' not in name and 'and' not in name:
            for key in self.tag_dict.keys():
                if key in name:
                    image_url = item['file_url']
                    folder_path = self.IMAGES_STORE_2 + '/' + folder
                    file_name = name + '.png'
                    # file_path = "{}/{}".format(folder_path, item['file_name'])
                    file_path = "{}/{}".format(folder_path, file_name)

                    req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
                    res = urllib.request.urlopen(req)
                    image = Image.open(res)
                    image.save(file_path)

                    self.csv_file_name_total = "{}/{}".format(self.FILES_STORE, 'pokemon_tag_'+ folder + '.csv')
                    with open(self.csv_file_name_total, 'a', newline='') as f:
                        self.tagWriter = csv.writer(f)
                        self.tagWriter.writerow(np.concatenate([[name], self.tag_dict[key]]))
                    break


    def process_photo(self, item):
        image_url = item['file_url']
        file_path = "{}/{}".format(self.IMAGES_STORE, item['file_name'])

        req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
        res = urllib.request.urlopen(req)
        image = Image.open(res)
        image.save(file_path)

    def Process_data(self, item):
        row = [item['idx'], item['name'], item['type1'], item['type2'], item['ability1'], item['ability2'], item['color']]
        with open(self.csv_file_name, 'a', newline='') as f:
            self.tagWriter = csv.writer(f)
            self.tagWriter.writerow(row)

    def process_item(self, item, spider):
        if isinstance(item, PokemonData):
            self.Process_data(item)
        elif isinstance(item, PokemonPhoto):
            self.process_photo(item)
        elif isinstance(item, PokemonPhoto2GetTag):
            self.process_photo_2tag(item)
        return item
