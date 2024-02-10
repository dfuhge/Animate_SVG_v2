#!usr/bin/python
# -*- coding: utf-8 -*-
import re
import os
import csv
from html.parser import HTMLParser
import requests
import warnings
warnings.filterwarnings("ignore")

# Root path for dataset
DATASET_ROOT = "results"
# Root path for Logo website
WEBSITE_ROOT = "https://worldvectorlogo.com/de/alphabetische/{}"
sub_categories = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPH_WEBSITES = [WEBSITE_ROOT.format(s) for s in sub_categories]

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By

# Initialize Main Firefox Driver
options = Options()
options.headless = True
driver = webdriver.Firefox(options=options)
print ("Headless Firefox Initialized")

# Collect SVG Links
svg_links = []

def is_not_found_page():
    html = driver.page_source
    return "Tut mir leid, wir konnten die Seite nicht finden, die Sie angefordert" in html

for website in ALPH_WEBSITES:
    driver.get(website)
    svg_img = driver.find_elements(By.CLASS_NAME, 'logo__img')
    for element in svg_img:
        src = element.get_attribute('src')
        svg_links.append(src)
    # Search all following sites
    # Get next site
    i = 2
    driver.get(website + '/' + str(i))
    while not is_not_found_page():
        print(website + '/' + str(i))
        svg_img = driver.find_elements(By.CLASS_NAME, 'logo__img')
        for element in svg_img:
            src = element.get_attribute('src')
            svg_links.append(src)
        i += 1
        driver.get(website + '/' + str(i))

with open("src/data_scraping/results.txt", "w") as file:
    for link in svg_links:
        file.write(link + "\n")
    file.close()
driver.quit()
