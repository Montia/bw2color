import PIL.Image as Image
import os
from tqdm import tqdm
import random
import argparse
from sketchKeras.main import get
from keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='./data/')
parser.add_argument('--save', default='./out/crop/')

mod = load_model('/home/u22520/mod.h5')
# mod = None


def stretch_img(path, shape=(256, 256)):
    img = Image.open(path)
    img = img.resize(shape, Image.ANTIALIAS)
    return img


def crop_img(path, shape=(256, 256)):
    img = Image.open(path)
    img = img.crop((0, 0, 219, 219))
    img = img.resize(shape, Image.ANTIALIAS)
    return img


def padding_to_512(img):
    new_img = Image.new('RGB', (512, 512), (255, 255, 255))
    new_img.paste(img, (0, 0))
    return new_img


def color2sketch_512(path):
    """ use sketchKeras project extract sketch """
    return Image.fromarray(get(path, '', mod))


def color2sketch_256(path):
    """
    use sketchKeras project extract sketch
    1. padding to 512x512 -> 2. get sketch -> 3. crop to 256x256
    """
    img_512 = padding_to_512(path)
    sketch = Image.fromarray(get(img_512, '', mod))
    return sketch.crop((0, 0, 256, 256))


# Combine image with its gray image horizontally [img, gray_img]
def color_with_gray(img: Image):
    gray_img = img.convert('L')
    new_img = Image.new('RGB', (512, 256))

    new_img.paste(img, (0, 0))
    new_img.paste(gray_img, (256, 0))

    return new_img


def resize(path, method):
    return method(path)


def preprocess(input, output, method, train_ratio=0.8):
    if not os.path.exists(output):
        os.makedirs(output)
        os.makedirs(output + 'train')
        os.makedirs(output + 'test')

    files = []
    for (dirpath, dirnames, filenames) in os.walk(input):
        if len(filenames) > 0 and dirpath != input:
            files.extend([dirpath + '/' + name for name in filenames])

    for image in tqdm(files):
        train = 'train/' if random.random() < train_ratio else 'test/'
        save_path = output + train + image.split('/')[-1]
        # try:
        # color_with_gray(resize(image, method)).save(save_path)
        color2sketch_256(resize(image, method)).save(save_path)
        # except:
        #     print('image %s is wrong' % image)


if __name__ == '__main__':
    a = parser.parse_args()
    data = a.data
    save = a.save
    preprocess(data, save, crop_img)
    # color2sketch_256(
    #     resize('C:/Users/Leeld/Documents/projects/bw2color/data/hyouka/1/thumb-350-286491.jpg', crop_img)).save('1.jpg')
