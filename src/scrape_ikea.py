from bs4 import BeautifulSoup
import time
import requests
import pandas as pd
# import csv
# from random import randint
# import sys


def flatten_list(my_list):
    '''flatten a nested list'''
    flat_list = []
    for ii in my_list:
        for jj in ii:
            flat_list.append(jj)

    return flat_list


headers = {"User-Agent": "Ubuntu Chromium/76.0.3809.100"}
# Mozilla/5.0 (X11; Linux x86_64)
# AppleWebKit/537.36 (KHTML, like Gecko)
# Chrome/76.0.3809.100 Safari/537.36}
ikea_base = 'https://www.ikea.com'
ikea_series = '/us/en/cat/series-series/'
url = ikea_base + ikea_series

r = requests.get(url, headers=headers)
soup = BeautifulSoup(r.text, 'lxml')

# list of all series
# find all series links
series_list = soup.find_all('li', {'class': 'range-catalog-list__item'})

series_link = []
for req in series_list:
    series_href = req.find_all('a', href=True)
    for sh in series_href:
        series_link.append(sh['href'])

# for each family get all products
all_product_link = []
all_product_img = []
for re in series_link:
    url = re + '?page=10'
    r = requests.get(url, headers=headers)
    series_soup = BeautifulSoup(r.text, 'lxml')
    item_list = series_soup.find_all('div',
                                     {'class': 'product-compact__spacer'})

    item_links = []
    img_links = []
    for il in item_list:
        item_href = il.find_all('a', href=True)
        for ih in item_href:
            item_links.append(ih['href'])
        img_links.append(il.img['src'])

    all_product_link.append(item_links)
    all_product_img.append(img_links)
    # breath
    time.sleep(1)

# unique-ness issues
all_product_link_temp = [list(set(apl)) for apl in all_product_link]

# flatten
all_product_link = flatten_list(all_product_link_temp)
# flatten
all_product_img = flatten_list(all_product_img)


all_product_desc = []
for re in all_product_link:
    url = re
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, 'lxml')

    product_html = soup.find_all('meta', {'itemprop': 'description'})

    product_desc = []
    # there is probably only one of these anyway
    for ppd in product_html:
        product_desc.append(ppd)

    all_product_desc.append(product_desc)

    # breath
    time.sleep(1)


# some times description is missing
temp = []
for apd in all_product_desc:
    if len(apd) == 0:
        temp.append('')
    else:
        temp.append(apd[0]['content'])

all_product_desc = temp


out_tuple = list(zip(all_product_link, all_product_img, all_product_desc))
out = pd.DataFrame(out_tuple, columns=['link', 'img_url', 'description'])

my_dir = '/home/peter/Documents/projects/insight/copyprisim/results/'
out.to_csv(my_dir + 'ikea.csv')
