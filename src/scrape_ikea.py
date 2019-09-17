from bs4 import BeautifulSoup
import time
import requests
import pandas as pd
# import csv
# from random import randint
# import sys

headers = {"User-Agent": "Ubuntu Chromium/76.0.3809.100"}
# Mozilla/5.0 (X11; Linux x86_64)
# AppleWebKit/537.36 (KHTML, like Gecko)
# Chrome/76.0.3809.100 Safari/537.36}
ikea_base = 'https://www.ikea.com'

# find the link to every family
all_families = []
for ii in range(0, 25):  # every letter in the alphabet
    url = ikea_base + '/us/en/catalog/productsaz/' + str(ii) + '/'
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, 'lxml')

    # the container for all product family links
    # some letters do not have product families -- use exception handling
    product_list = soup.find_all("li", {"class": "productsAzLink"})

    # extract link to each family-s splash page
    # if there aren't any products on that page, skip to next iteration
    if(len(product_list) == 0):
        continue
    else:
        family_links = []
        for req in product_list:
            family_hrefs = req.find_all("a", href=True)
            for link_element in family_hrefs:
                family_links.append(link_element['href'])
        all_families.append(family_links)


# all_families is a list of lists
# list elements are each a list of family links
# flatten this list
all_families = [val for sublist in all_families for val in sublist]

# for each family get all products
all_products = []
for fl in all_families:
    url = ikea_base + fl
    r = requests.get(url, headers=headers)
    family_soup = BeautifulSoup(r.text, 'lxml')

    item_list = family_soup.find_all("div", {"class": "productLists"})

    item_links = []  # links to each product
    for req in item_list:
        item_hrefs = req.find_all("a", {"class": "productLink"}, href=True)
        for link_element in item_hrefs:
            item_links.append(link_element['href'])

    all_products.append(item_links)


# family_item_links is a list of lists
# list elements are each a list of product links
# flatten this list
all_products = [val for sublist in all_products for val in sublist]

my_dir = '/home/peter/Documents/projects/insight/copyprisim/results/'
info_picture = []
info_corpus = []
i = 0
for pp in all_products:
    url = ikea_base + pp
    r = requests.get(url, headers=headers)
    product_soup = BeautifulSoup(r.text, 'lxml')

    # image of product
    try:
        product_img = product_soup.find("div",
                                        {"class": "rightContentContainer"}).img
        product_img = product_img['src']
    except AttributeError:
        product_img = ''

    info_picture.append(product_img)

    # text description information
    product_info = product_soup.find_all('div',
                                         {"id": "custBenefit"},
                                         {"class": "texts keyFeaturesmargin"})

    info_list = []
    for req in product_info:
        list_item = req.find_all('div')
        for li in list_item:
            info_list.append(li.get_text())

    info_list = [i.replace('-', '') for i in info_list]
    info_corpus.append(' '.join(info_list))
    # i = i + 1
    # if i % 100 == 0:
    #     out = pd.DataFrame(list(zip(info_picture, info_corpus)),
    #                        columns=['img_url', 'description'])
    #     out.to_csv(my_dir + 'writing.csv')

    # breath
    time.sleep(1)


output = pd.DataFrame(list(zip(info_picture, info_corpus)),
                      columns=['img_url', 'description'])

output.to_csv(my_dir + 'ikea.csv')
