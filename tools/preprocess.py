import PIL.Image as Image
import os
from tqdm import tqdm
import random
import argparse

""" input params """
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='./data/')
parser.add_argument('--save', default='./out/crop/')
# t - use sketch model, f - use basic gray
parser.add_argument('--method', default='gray')
parser.add_argument('--mod_path', default='/home/u22520/mod.h5')


def stretch_img(img, shape=(256, 256)):
    img = img.resize(shape, Image.ANTIALIAS)
    return img


def crop_img(img, shape=(256, 256)):
    #img = img.crop((0, 0, 219, 219))
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


def joint256img(left, right):
    new_img = Image.new('RGB', (512, 256))
    new_img.paste(left, (0, 0))
    new_img.paste(right, (256, 0))
    return new_img

def joint512img(left, right):
    new_img = Image.new('RGB', (1024, 512))
    new_img.paste(left, (0, 0))
    new_img.paste(right, (512, 0))
    return new_img

def join_img(left, right):
    assert(left.size == right.size)
    width = left.size[0] * 2
    height = left.size[1]
    new_img = Image.new('RGB', (width, height))
    new_img.paste(left, (0, 0))
    new_img.paste(right, (left.size[0], 0))
    return new_img

# Combine image with its gray image horizontally [img, gray_img]
def color_with_gray(img: Image):
    return join_img(img, img.convert('L'))


def color_with_sketch(img: Image):
    return joint256img(img, color2sketch_256(img))


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
        sub_dir = 'train/' if random.random() < train_ratio else 'test/'
        save_path = output + sub_dir + image.split('/')[-1]
        '''try:
            method(image).save(save_path)
        except:
            print('image %s is wrong' % image)'''
        method(image).save(save_path)


# ============================================================================
# preprocess functions
# name rule: [resize method]2[target]
# input:    image path
# output:   PIL image object

def crop2gray(path):
    img = Image.open(path)
    return color_with_gray(img)


def crop2sketch(path):
    img = Image.open(path)
    return color_with_sketch(crop_img(img))


if __name__ == '__main__':
    a = parser.parse_args()
    data = a.data
    save = a.save
    method = a.method
    mod_path = a.mod_path

    if save[-1] != '/':
        save += '/'

    process_method = crop2gray
    if method == 'sketch':
        process_method = crop2sketch
        mod = load_model(a.mod_path)

    print('use %s method to process images in %s to %s' % (method, data, save))
    preprocess(data, save, process_method)
