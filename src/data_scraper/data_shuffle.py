import random

random.seed(0)

def shuffle_links():
    links = {}
    with open('src/data_scraper/results.txt') as f:
        i = 0
        for line in f:
            links[i] = line
            i += 1
    filenames = list(links.keys())
    random.shuffle(filenames)
    with open('src/data_scraper/results_shuffled.txt', 'w') as f:
        for i in range(0, len(filenames)):
            f.write(f'logo_{str(i)}.svg;{links[filenames[i]]}')


shuffle_links()