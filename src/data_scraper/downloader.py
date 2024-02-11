#!usr/bin/python
# -*- coding: utf-8 -*-
import random
import os
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append(os.getcwd())

import urllib.request

# Seed random
random.seed(0)

DATASET_ROOT = "data/raw_dataset"
logo_id = 0

def download_image_from_link(link):
    global logo_id
    format = link.split('.')[3][:3]
    if "svg" in format:
        # retrieve content
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        content = urllib.request.urlopen(link).read().decode('UTF-8')
        if not os.path.exists(DATASET_ROOT):
            os.mkdir(DATASET_ROOT)
        with open(os.path.join(DATASET_ROOT, 'logo_') + str(logo_id) + '.' + format, 'w') as file:
            file.write(content)
        logo_id += 1


def download_images(img_numbers):
    with open('src/data_scraper/results.txt') as f:
        i = 0
        for line in f:
            if i in img_numbers:
                download_image_from_link(line)
            i += 1

# Adjust these numbers to get a different range and size of samples
img_list = random.sample(range(0, 119820), 2000)
download_images(img_list)
