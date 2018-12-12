import numpy as np
from PIL import Image
import tensorflow as tf

def get_tfrecord(batch_size, training):
    #返回X，Y_real， 形状为[batch_size, 256, 256, 3]， 值为-1到1
    img = []
    img_bw = []
    for i in range(batch_size):
        t = Image.fromarray((np.random.random([256, 256, 3]) * 256).astype(np.uint8))
        img.append(np.array(t))
        img_bw.append(np.array(t.convert('L').convert('RGB')))
    return tf.Variable(np.array(img_bw) / 128 - 1), tf.Variable(np.array(img) / 128 - 1)