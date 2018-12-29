import tensorflow as tf #引入tensorflow

KERNEL_SIZE = 3 #生成器卷积核长宽为3
FIRST_OUTPUT_CHANNEL = 64 #生成器encoder
MAX_OUTPUT_CHANNEL_LAYER = 5 #通道数在第五层后不再增长，之前每次卷积提升两倍
REGULARIZER = 0 #权重衰减正则化系数为0，不使用
DROPOUT = 0.5 #dropout率为0.5

def get_weight(shape, regularizer=None):#获取w的函数，与老师的代码相同，解释略
    w = tf.Variable(tf.random_normal(shape, stddev=0.5))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def gen_conv(X, kernels, stride=2, regularizer=None):#获取生成器卷积层的函数
    w = get_weight([KERNEL_SIZE, KERNEL_SIZE, X.get_shape().as_list()[-1], kernels], regularizer)#获取卷积核
    return tf.nn.conv2d(X, w, strides=[1, stride, stride, 1], padding='SAME')#用得到的卷积核以及给定的步长建立卷积层

def gen_deconv(X, kernels, batch_size, stride=2, regularizer=None):#获取生成器反卷积层的函数
    w = get_weight([KERNEL_SIZE, KERNEL_SIZE, kernels, X.get_shape().as_list()[-1]], regularizer)#获取卷积核
    input_shape = tf.shape(X)#获取该层输入的形状
    output_shape = [input_shape[0], input_shape[1] * stride, input_shape[2] * stride, kernels]#求得该层输出的形状
    return tf.nn.conv2d_transpose(X, w, output_shape=output_shape, strides=[1, stride, stride, 1], padding='SAME')#用得到的卷积核、输出形状以及步长构建反卷积层

def batchnorm(inputs):#批标准化的函数
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))#对输入进行批标准化

def lrelu(x, a=0.2):#leak relu，跟relu相似的激活函数，区别在于赋予了负数一点梯度
    return ((1 + a) * x + (1 - a) * tf.abs(x)) / 2#计算lrelu，对于每一个分量x，等价于x if x >= 0 else a * x

def forward(X, batch_size, training):#前向传播函数定义了GAN的生成器，用了U-net的结构
    layers = [X]#layers用来存储每一模块的结果，X被看作模块0的结果
    #Encoder
    for i in range(6):#encoder共6个模块，每个模块内含两层卷积，第一层步长为一，第二层步长为二
        convolved1 = gen_conv(layers[-1], layers[-1].get_shape().as_list()[-1], 1)#先是一个步长为一的卷积
        normed1 = batchnorm(convolved1)#进行批标准化
        output1 = lrelu(normed1)#通过激活函数
        convolved2 = gen_conv(tf.concat([layers[-1], output1], axis=3), FIRST_OUTPUT_CHANNEL * 2 ** min(MAX_OUTPUT_CHANNEL_LAYER, i))#再来一个卷积，这个卷积连接了前两个形状一样的层
        normed2 = batchnorm(convolved2)#进行批标准化
        output2 = lrelu(normed2)#通过激活函数
        layers.append(output2)#将该模块儿结果加入layers

    #Decoder
    for i in range(5):#decoder也是六个模块儿，这个循环构建前五个，在与encoder相反的基础上，增加了与encoder对应层的连接（skip connection)
        convolved1 = gen_deconv(layers[-1], layers[-1].get_shape().as_list()[-1], batch_size, 1)#先是一个步长为一的反卷积
        normed1 = batchnorm(convolved1)#通过批标准化层
        output1 = lrelu(normed1)#再通过激活函数
        skip_layer = 6 - i#求出在encoder中的对应层
        if i == 0:#如果是decoder的第一层
            deconvolved2 = gen_deconv(tf.concat([layers[-1], output1], axis=3), FIRST_OUTPUT_CHANNEL * 2 ** min(MAX_OUTPUT_CHANNEL_LAYER, 4 - i), batch_size)#对应的层就是自己，不用管
        else:#如果不是的话
            deconvolved2 = gen_deconv(tf.concat([layers[-1], output1, layers[skip_layer]], axis=3), FIRST_OUTPUT_CHANNEL * 2 ** min(MAX_OUTPUT_CHANNEL_LAYER, 4 - i), batch_size)#既要连接前两层，又要连接encoder的对应层
        output2 = batchnorm(deconvolved2)#通过批标准化层
        if i < 2:#在decoder的前两层
            output2 = tf.nn.dropout(output2, 1 - DROPOUT)#通过dropout提供噪声，取代GAN中的z，防止生成单一图片
        output2 = lrelu(output2)#通过激活函数
        layers.append(output2)#将模块结果加入layers
    convolved1 = gen_deconv(layers[-1], layers[-1].get_shape().as_list()[-1], batch_size, 1)#最后一个模块的第一个反卷积
    normed1 = batchnorm(convolved1)#批标准化
    output1 = lrelu(normed1)#激活函数
    convolved2 = gen_deconv(tf.concat([layers[-1], output1, layers[1]], axis=3), 3, batch_size)#最后一个模块的第二个反卷积
    output2 = tf.nn.tanh(convolved2)#激活函数tanh将输出层结果限制在-1到1
    layers.append(output2)#将最后的结果加入layers
    if training == True:#在训练时
        return layers[-1], layers[6]#返回生成器的输出，并将中间层提供给guide decoder
    else:#在测试和应用中
        return layers[-1]#只返回生成器的输出
    

        
