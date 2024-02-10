#!usr/bin/python
# -*- coding: utf-8 -*-
import re
import random
import os
import csv
from html.parser import HTMLParser
import requests
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append(os.getcwd())

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import WebDriverException
from selenium.common.exceptions import StaleElementReferenceException

# Seed random
random.seed(0)

DATASET_ROOT = "data/raw_dataset"
logo_id = 0

# Initialize Main Firefox Driver



def download_image_from_link(link):
    global logo_id
    format = link.split('.')[3][:3]
    if "svg" in format:
        # retrieve content
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0'}
        response = requests.get(link, headers=headers)
        # save
        with open(os.path.join(DATASET_ROOT, 'logo_') + str(logo_id) + '.' + format, "wb") as svg:
            svg.write(response.content)
        print("successfully downloaded " + link)
        logo_id += 1

def download_images(img_numbers):
    with open('src/data_scraper/results.txt') as f:
        i = 0
        for line in f:
            if i in img_numbers:
                download_image_from_link(line)
            i += 1

# Adjust these numbers to get a different range and size of samples
img_list = random.sample(range(0, 119820), 5000)
download_images(img_list)
