import tensorflow as tf #引入tensorflow

KERNEL_SIZE = 3 #生成器卷积核长宽为3
FIRST_OUTPUT_CHANNEL = 64 #生成器encoder
MAX_OUTPUT_CHANNEL_LAYER = 5 #通道数在第五层后不再增长，之前每次卷积提升两倍
REGULARIZER = 0 #权重衰减正则化系数为0，不使用
DROPOUT = 0.5 #dropout率为0.5

def get_weight(shape, regularizer=None):#获取w，与老师的代码相同，略
    w = tf.Variable(tf.random_normal(shape, stddev=0.5))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def gen_conv(X, kernels, stride=2, regularizer=None):#获取生成器的卷积层
    w = get_weight([KERNEL_SIZE, KERNEL_SIZE, X.get_shape().as_list()[-1], kernels], regularizer)#获取卷积核
    return tf.nn.conv2d(X, w, strides=[1, stride, stride, 1], padding='SAME')#用得到的卷积核以及给定的步长建立卷积层

def gen_deconv(X, kernels, batch_size, stride=2, regularizer=None):
    w = get_weight([KERNEL_SIZE, KERNEL_SIZE, kernels, X.get_shape().as_list()[-1]], regularizer)
    input_shape = tf.shape(X)
    output_shape = [input_shape[0], input_shape[1] * stride, input_shape[2] * stride, kernels]
    return tf.nn.conv2d_transpose(X, w, output_shape=output_shape, strides=[1, stride, stride, 1], padding='SAME')

def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

def lrelu(x, a=0.2):
    return ((1 + a) * x + (1 - a) * tf.abs(x)) / 2

def forward(X, batch_size, training):
    #X的形状为[None, 512, 512, 3], 值为-1到1
    layers = [X]
    #Encoder
    for i in range(6):
        convolved1 = gen_conv(layers[-1], layers[-1].get_shape().as_list()[-1], 1)
        normed1 = batchnorm(convolved1)
        output1 = lrelu(normed1)
        convolved2 = gen_conv(tf.concat([layers[-1], output1], axis=3), FIRST_OUTPUT_CHANNEL * 2 ** min(MAX_OUTPUT_CHANNEL_LAYER, i))
        normed2 = batchnorm(convolved2)
        output2 = lrelu(normed2)
        layers.append(output2)

    #Decoder
    for i in range(5):
        convolved1 = gen_deconv(layers[-1], layers[-1].get_shape().as_list()[-1], batch_size, 1)
        normed1 = batchnorm(convolved1)
        output1 = lrelu(normed1)
        skip_layer = 6 - i
        if i == 0:
            deconvolved2 = gen_deconv(tf.concat([layers[-1], output1], axis=3), FIRST_OUTPUT_CHANNEL * 2 ** min(MAX_OUTPUT_CHANNEL_LAYER, 4 - i), batch_size)
        else:
            deconvolved2 = gen_deconv(tf.concat([layers[-1], output1, layers[skip_layer]], axis=3), FIRST_OUTPUT_CHANNEL * 2 ** min(MAX_OUTPUT_CHANNEL_LAYER, 4 - i), batch_size)
        output2 = batchnorm(deconvolved2)
        if i < 2 and training:
            output2 = tf.nn.dropout(output2, 1 - DROPOUT)
        output2 = lrelu(output2)
        layers.append(output2)
    convolved1 = gen_deconv(layers[-1], layers[-1].get_shape().as_list()[-1], batch_size, 1)
    normed1 = batchnorm(convolved1)
    output1 = lrelu(normed1)
    output2 = gen_deconv(tf.concat([layers[-1], output1, layers[1]], axis=3), 3, batch_size)
    output2 = tf.nn.tanh(output2)
    layers.append(output2)
    if training == True:
        return layers[-1], layers[6]
    else:
        return layers[-1]
    

        
