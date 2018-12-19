import tensorflow as tf
import numpy as np
import keras
from PIL import Image
from myvgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

img = Image.open('data/hyouka/1/thumb-350-286491.jpg')
img = img.crop((0, 0, 219, 219))
img = img.resize((224, 224), Image.ANTIALIAS)


def test():
    tf.reset_default_graph()

    my_img = tf.placeholder(tf.float32, (1, 224, 224, 3), name='my_original_image')
    vgg = VGG19(image_shape=(1, 224, 224, 3), input_tensor=my_img)

    output = tf.identity(vgg['block5_pool'], name='my_output')

    with tf.Session() as sess:
        vgg.load_weights()
        fd = {my_img: np.expand_dims(img, 0)}

        output_val = sess.run(output, fd)

    print('')
    print(output_val.shape, output_val.mean())


def original():
    vgg19 = keras.applications.VGG19(weights='imagenet', include_top=False)
    img_input = np.expand_dims(img, 0)

    img_input = preprocess_input(img_input)
    keras_output = vgg19.predict(img_input)
    print('keras')
    print(keras_output.shape, keras_output.mean())


if __name__ == '__main__':
    original()
    test()
