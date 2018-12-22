import tensorflow as tf
from PIL import Image
import numpy as np

KERNEL_SIZE = 3
STRIDE = 1
DESTRIDE = 2
FIRST_OUTPUT_CHANNEL = 8
REGULARIZER = 0
DROPOUT = 0.5


def get_weight(shape, regularizer=None):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.2))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def gen_conv(X, channel, kernels, regularizer=None):
    channel = int(channel)
    kernels = int(kernels)
    w = get_weight([KERNEL_SIZE, KERNEL_SIZE, channel, kernels], regularizer)
    return tf.nn.conv2d(X, w, strides=[1, STRIDE, STRIDE, 1], padding='SAME')


def gen_deconv(X, channel, kernels, batch_size, regularizer=None):
    channel = int(channel)
    kernels = int(kernels)
    w = get_weight([KERNEL_SIZE, KERNEL_SIZE, channel, kernels], regularizer)
    output_shape = X.get_shape().as_list()
    output_shape[0] = batch_size
    output_shape[1] *= 2
    output_shape[2] *= 2
    output_shape[3] = channel
    return tf.nn.conv2d_transpose(X, w, output_shape=output_shape, strides=[1, DESTRIDE, DESTRIDE, 1], padding='SAME')


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True,
                                         gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def max_pool(input):
    pool = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    return tf.nn.dropout(x=pool, keep_prob=DROPOUT)


def lrelu(x, a=0.5):
    return ((1 + a) * x + (1 - a) * tf.abs(x)) / 2


def skip_connect(encoder_result, upsample_result):
    return tf.concat(values=[encoder_result, upsample_result], axis=-1)


def double_conv(X, channel, kernel):
    # print(layers[-1].shape)
    conv1 = gen_conv(X, channel, kernel)
    # todo check batch norm function
    norm1 = batchnorm(conv1)
    relu1 = tf.nn.relu(norm1)
    # print(relu1.shape)

    conv2 = gen_conv(relu1, kernel, kernel)
    norm2 = batchnorm(conv2)
    relu2 = tf.nn.relu(norm2)
    # print(relu2.shape)
    return relu2


def upsample(X, channel, kernel, batch_size):
    up = gen_deconv(X, channel, kernel, batch_size)
    norm = batchnorm(up)
    relu = tf.nn.relu(norm)
    dropout = tf.nn.dropout(x=relu, keep_prob=DROPOUT)
    return dropout


def forward(X, batch_size, training):
    # X的形状为[None, 512, 512, 1], 值为-1到1

    layers = [batchnorm(X)]
    skip_layers = []
    # 4 layer Encoder: double conv + max pool
    for i in [64, 128, 256, 512]:
        # print(layers[-1].shape)
        # double conv
        channel = 1 if i == 64 else i / 2
        relu2 = double_conv(layers[-1], channel, i)
        skip_layers.append(relu2)

        # max pool
        pool = max_pool(relu2)
        # print("")
        # print(pool.shape)
        layers.append(pool)

    # bottom layer: double conv + unsample
    relu2 = double_conv(layers[-1], 512, 1024)

    # print("")
    sample = upsample(relu2, 512, 1024, batch_size)
    layers.append(sample)

    # 4 level Decoder: joint + double conv + upsample
    for i in [512, 256, 128]:
        joint = skip_connect(skip_layers[-1], layers[-1])
        skip_layers.pop()

        relu2 = double_conv(joint, i * 2, i)

        # todo
        sample = upsample(relu2, i / 2, i, batch_size)
        layers.append(sample)

    i = 64
    joint = skip_connect(skip_layers[-1], layers[-1])
    skip_layers.pop()

    relu2 = double_conv(joint, i * 2, i)

    # todo
    sample = gen_conv(relu2, i, 1)
    norm3 = batchnorm(sample)

    return norm3


def test_forward():
    sketch = Image.open('U-net/data_set/train/1.tif').convert('L')

    X = tf.placeholder(tf.float32, (1, 512, 512, 1))
    result = forward(X, 1, True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        img = sess.run(result, feed_dict={X: np.expand_dims(sketch, 0).reshape(1, 512, 512, 1)})

        print(img.shape)
        img = img.reshape([512, 512, 2])
        Image.fromarray(img.astype(np.uint8)).save('1.tif')


if __name__ == '__main__':
    test_forward()
