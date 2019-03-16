# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class PokemonData(scrapy.Item):
    # define the fields for your item here like:
    idx = scrapy.Field()
    name = scrapy.Field()
    type1 = scrapy.Field()
    type2 = scrapy.Field()
    ability1 = scrapy.Field()
    ability2 = scrapy.Field()
    color = scrapy.Field()
    pass

class PokemonPhoto(scrapy.Item):
    file_name = scrapy.Field()
    file_url = scrapy.Field()
    pass

class PokemonPhoto2GetTag(scrapy.Item):
    pokemon_name = scrapy.Field()
    file_name = scrapy.Field()
    file_url = scrapy.Field()
    size = scrapy.Field()
    artwork = scrapy.Field()
    pass
    
