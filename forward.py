import tensorflow as tf 

KERNEL_SIZE = 4
STRIDE = 2
FIRST_OUTPUT_CHANNEL = 8
MAX_OUTPUT_CHANNEL_LAYER = 8
REGULARIZER = 0
DROPOUT = 0.5

def get_weight(shape, regularizer=None):
    w = tf.Variable(tf.random_normal(shape, stddev=0.5))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def gen_conv(X, kernels, regularizer=None):
    w = get_weight([KERNEL_SIZE, KERNEL_SIZE, X.get_shape().as_list()[-1], kernels], regularizer)
    return tf.nn.conv2d(X, w, strides=[1, STRIDE, STRIDE, 1], padding='SAME')

def gen_deconv(X, kernels, batch_size, regularizer=None):
    w = get_weight([KERNEL_SIZE, KERNEL_SIZE, kernels, X.get_shape().as_list()[-1]], regularizer)
    output_shape = X.get_shape().as_list()
    output_shape[0] = batch_size
    output_shape[1] *= 2
    output_shape[2] *= 2
    output_shape[3] = kernels
    return tf.nn.conv2d_transpose(X, w, output_shape=output_shape, strides=[1, STRIDE, STRIDE, 1], padding='SAME')

def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

def lrelu(x, a=0.5):
    return ((1 + a) * x + (1 - a) * tf.abs(x)) / 2

def forward(X, batch_size, training):
    #X的形状为[None, 512, 512, 3], 值为-1到1
    layers = [X]
    #Encoder
    for i in range(9):
        convolved = gen_conv(layers[-1], FIRST_OUTPUT_CHANNEL * 2 ** min(MAX_OUTPUT_CHANNEL_LAYER, i))
        normed = batchnorm(convolved)
        output = lrelu(normed)
        layers.append(output)
    #return layers[9]

    #Decoder
    for i in range(8):
        skip_layer = 9 - i
        if i == 0:
            deconvolved = gen_deconv(layers[-1], FIRST_OUTPUT_CHANNEL * 2 ** min(MAX_OUTPUT_CHANNEL_LAYER, 7 - i), batch_size)
        else:
            deconvolved = gen_deconv(tf.concat([layers[-1], layers[skip_layer]], axis=3), FIRST_OUTPUT_CHANNEL * 2 ** min(MAX_OUTPUT_CHANNEL_LAYER, 7 - i), batch_size)
        output = batchnorm(deconvolved)
        if i < 3 and training:
            output = tf.nn.dropout(output, 1 - DROPOUT)
        output = lrelu(output)
        layers.append(output)
    output = gen_deconv(tf.concat([output, layers[1]], axis=3), 3, batch_size)
    output = tf.nn.tanh(output)
    layers.append(output)
    if training == True:
        return layers[-1], layers[9]
    else:
        return layers[-1]
    

        
