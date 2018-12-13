import requests as rq
from bs4 import BeautifulSoup
import os
import PIL.Image as image
import PIL

# Global configurations
url = 'https://wall.alphacoders.com/search.php?search=%s&lang=Chinese&page=%d'
mozilla_header = {"User-Agent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.10; rv:41.0) Gecko/20100101 Firefox/41.0', }
PATH = './data/'
ANIME_LIST = ['your+name', 'love+live', 'The+Melancholy+Of+Haruhi+Suzumiya', 'hyouka', 'Sound+Euphonium']


def download_one_image(path, url):
    print('downloading %s to %s' % (url, path))
    with open(path, 'wb') as f:
        f.write(rq.get(url, headers=mozilla_header).content)


def download_one_page(name, page, page_path):
    print('downloading %s image in page %d' % (name, page))
    r = rq.get(url % (name, page), headers=mozilla_header)
    soup = BeautifulSoup(r.text, 'html.parser')

    divs = soup.find_all('div', {'class': 'thumb-container-big'})
    for div in divs:
        try:
            img = div.find_all('img')[0]
            img_url = img['src']
            img_name = img_url.split('/')[-1]
            download_one_image(page_path + img_name, img_url)
        except:
            print('downloading %s.png fail' % div[id])

    # get page count
    if page == 1:
        ul = soup.find_all('ul', {'class': 'pagination'})[-1]
        return int(ul.find_all('li')[-2].text)


def get_page(name):
    r = rq.get(url % (name, 1), headers=mozilla_header)
    soup = BeautifulSoup(r.text, 'html.parser')
    ul = soup.find_all('ul', {'class': 'pagination'})[-1]
    return int(ul.find_all('li')[-2].text)


def download_anime(name):
    # make name -> page folder structure
    print('downloading anime:', name)
    if not os.path.exists(PATH + name):
        os.mkdir(PATH + name)
    page = get_page(name)
    for i in range(page):
        page_path = PATH + name + '/' + str(i + 1) + '/'
        if os.path.exists(page_path):
            print('%s page %d exists' % (name, page))
            continue
        else:
            os.mkdir(page_path)
            download_one_page(name, i, page_path)


def download_wallpaper():
    for name in ANIME_LIST:
        download_anime(name)


if __name__ == '__main__':
    download_wallpaper()
