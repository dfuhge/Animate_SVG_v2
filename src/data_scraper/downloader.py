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

def download_image_from_link(filename, link):
    global logo_id
    # retrieve content
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    try:
        content = urllib.request.urlopen(link).read().decode('UTF-8')
        if not os.path.exists(DATASET_ROOT):
            os.mkdir(DATASET_ROOT)
        with open(os.path.join(DATASET_ROOT, filename), 'w') as file:
            file.write(content)
    except:
        print('Could not download: ' + link)
    logo_id += 1


def download_images(scope):
    with open('src/data_scraper/results_shuffled.txt') as f:
        i = 0
        for line in f:
            if i < scope:
                s = line.split(';')
                download_image_from_link(s[0], s[1])
            i += 1

# Adjust these numbers to get a different range and size of samples
download_images(2000)
