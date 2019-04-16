from scrapy import Spider, Request
from pokeball.items import PokemonData, PokemonPhoto, PokemonPhoto2GetTag


class pokeball(Spider):
    name = "pokeball"

    allowed_domains = ['https://bulbapedia.bulbagarden.net', ]

    start_urls = [
        'https://bulbapedia.bulbagarden.net/wiki/Bulbasaur_(Pok%C3%A9mon)']

    def parse_img(self, response):
        image_url_final = response.xpath(
            '//div[@class="fullImageLink"]/a/@href').get()
        file_name = image_url_final.split('/')[-1]
        image_url_final = "https:%s" % image_url_final

        yield PokemonPhoto(file_url=image_url_final, file_name=file_name)

    # def parse(self, response):
    #     filename = 'test.html'
    #     with open(filename, 'wb') as f:
    #         f.write(response.body)

    def parse(self, response):
        # tag
        name = response.xpath(
            '//h1[@id="firstHeading"]/text()').get().split()[0]
        # num = response.xpath(
        #     '//div[@id="mw-content-text"]/table[1]/tr[2]/td[3]/table/tr/td[1]/a/span/text()').get().split()[0][:-1]
        num = response.xpath(
            '//a[contains(@title, "List of Pokémon by National Pokédex number")]/span/text()').getall()[1]
        types = response.xpath(
            '//a[contains(@title, "(type)")][contains(@href, "wiki")]/span/b/text()').getall()[0:2]
        len_table = len(response.xpath(
            '//table[contains(@style, "float:right;")]/tr').getall())
        color = response.xpath(
            '//table[contains(@style, "float:right;")]/tr[' + str(len_table-1) + ']/td[1]/table/tr/td/text()').getall()[1].split()[0]
        """
            response.xpath(
                            '//table[contains(@style, "float:right;")]/tr[12]/td[1]/table/tr/td/text()').getall()[1]
            response.xpath(
                            '//table[contains(@style, "float:right;")]/tr[12]/td[1]/table/tr/td/text()').getall()[1].split()[0]
        """
        abilities = response.xpath(
            '//a[contains(@href, "(Ability)")]/span/text()').getall()[0:2]

        yield PokemonData(idx=num, name=name, type1=types[0], type2=types[1], ability1=abilities[0], ability2=abilities[1], color=color)

        # image
        image = response.xpath(
            '//table[contains(@style, "float:right;")]/tr[1]/td/table/tr[2]/td/table/tr[1]/td/a/@href').get()
        image_url = "{}{}".format(self.allowed_domains[0], image)

        # yield Request(image_url, callback=self.parse_img, dont_filter=True)

        # follow link
        next_num = response.xpath(
            '//div[@id="mw-content-text"]/table[1]/tr/td/table/tr/td[contains(@style, "left")][1]/a/span/text()').get().split()[0][:-1]

        if next_num != '#???':
            next_url_relative = response.xpath(
                '//div[@id="mw-content-text"]/table[1]/tr/td/table/tr/td[contains(@style, "left")]/a/@href').get()
            next_url = self.allowed_domains[0] + next_url_relative

            yield Request(next_url, callback=self.parse, dont_filter=True)


class masterball(Spider):
    name = 'masterball'

    allowed_domains = ['https://archives.bulbagarden.net',]

    def start_requests(self):
        urls = [
            'https://archives.bulbagarden.net/wiki/Category:Ken_Sugimori_Pok%C3%A9mon_artwork',
            'https://archives.bulbagarden.net/wiki/Category:Pok%C3%A9mon_Dream_World_artwork',
            'https://archives.bulbagarden.net/wiki/Category:Official_anime_artwork',
        ]
        artwork_types = ['ken sugimori', 'dream work', 'os anime']

        for t, url in zip(artwork_types, urls):
            yield Request(url=url, callback=self.parse, meta={'artwork': t})
        # yield Request(url=urls[0], callback=self.parse, meta={'artwork': artwork_types[0]})
        self.count  = 0

    def parse_img(self, response):
        image_url_final = response.xpath(
            '//div[@class="fullImageLink"]/a/@href').get()
        # image_fise_size = response.xpath(
        #     '//div[@class="fullMedia"]/span/text()').get()
        file_name = image_url_final.split('/')[-1]
        name = file_name.split('.')[0]
        image_url_final = "https:%s" % image_url_final

        yield PokemonPhoto2GetTag(pokemon_name=name, file_url=image_url_final, file_name=file_name, size=response.meta['size'], artwork=response.meta['artwork'])

    def parse(self, response):
        list_img_rel_path = response.xpath(
            '//div[@id="mw-category-media"]/ul[@class="gallery mw-gallery-traditional"]/li/div/div[@class="gallerytext"]/a/@href').getall()
        list_img_size = response.xpath(
            '//div[@id="mw-category-media"]/ul[@class="gallery mw-gallery-traditional"]/li/div/div[@class="gallerytext"]/text()').getall()

        for i, rel_path in enumerate(list_img_rel_path):
            if True or i < 2:
                img_size_info = list_img_size[(i + 1) * 3 - 2].split()
                if img_size_info[1] == 'MB':
                    img_size = round(float(img_size_info[0])* 1024, 2)
                elif img_size_info[1] == 'KB':
                    img_size = 1
                    if ',' in img_size_info[0]:
                        img_size = float(img_size_info[0].replace(',', ''))
                    else:
                        img_size = float(img_size_info[0])
                else:
                    img_size = 1
                next_url = self.allowed_domains[0] + rel_path
                yield Request(url=next_url, callback=self.parse_img, dont_filter=True, meta={'size': img_size, 'artwork': response.meta['artwork']})
        
        next_page_urls = response.xpath(
            '//div[@id="mw-category-media"]/a/@href').getall()
        urls_len = len(next_page_urls)
        is_end = False
        if urls_len == 2:
            is_end = 'until' in next_page_urls[0]

        if not is_end:
            if len(next_page_urls) == 2:
                next_page_url = next_page_urls[0]
            elif len(next_page_urls) == 4:
                next_page_url = next_page_urls[1]

            next_page_url = self.allowed_domains[0] + next_page_url
            # print(next_page_url)
            yield Request(url=next_page_url, callback=self.parse, dont_filter=True, meta=response.meta)
                