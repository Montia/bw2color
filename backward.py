import numpy as np #这部分引入需要的模块
from PIL import Image
import tensorflow as tf
from time import sleep
import os
from time import sleep
import generateds
from tqdm import tqdm, trange
import forward

BATCH_SIZE = 1#一个batch的大小
L1_WEIGHT = 50#生成器损失中l1距离的权重
GAN_WEIGHT = 1#生成器损失中GAN提供损失的权重
GUIDE_DECODER_WEIGHT = 1#生成器损失中guide decoder与原图l1距离所占权重
EPS = 1e-12#防止log里是零的小量
LEARNING_RATE = 0.0002#学习率
BETA1 = 0.5#Adam的参数
EMA_DECAY = 0.98#滑动平均的衰减率
TOTAL_STEP = 50000 #训练轮数
PARAMS = 'l1weight={},gfc={}, mcl={} with guide decoder'.format(L1_WEIGHT, forward.FIRST_OUTPUT_CHANNEL, forward.MAX_OUTPUT_CHANNEL_LAYER)#目录名中的超参数
MODEL_SAVE_PATH = 'model_{}'.format(PARAMS)#模型存储目录
MODEL_NAME = 'pix2pix_model'#模型存档名
TRAINING_RESULT_PATH = 'training_result_{}'.format(PARAMS)#存储展示训练成果的图的目录名
GUIDE_DECODER_PATH = 'guide_decoder_{}'.format(PARAMS)#存储guide decoder产生图片的目录名
SAVE_FREQ = 1000#保存模型的频率
DISPLAY_FREQ = 100#展示训练效果的频率
DISPLAY_GUIDE_DECODER_FREQ = 100 #展示guide decoder生成图片的频率

def backward():#反向传播模块，包括了GAN的判别器、guide decoder以及与模型训练相关的操作
    def dis_conv(X, kernels, stride, layer, regularizer=None):#生成反卷积层的函数
        w = tf.get_variable('w{}'.format(layer), [forward.KERNEL_SIZE, forward.KERNEL_SIZE, X.get_shape().as_list()[-1], kernels], initializer=tf.truncated_normal_initializer(0, 0.2))#获取卷积核
        padded_X = tf.pad(X, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')#手动padding一波
        return tf.nn.conv2d(padded_X, w, [1, stride, stride, 1], padding='VALID')#用刚得到的卷积核以及步长构建卷积层

    def discriminator(discriminator_input, discriminator_output):#定义GAN的判别器
        X = tf.concat([discriminator_input, discriminator_output], axis=3)#将黑白图与彩色图摞在一起作为输入
        layers = [X]#layers用来存储各层结果
        for i in range(6):#判别器包括六层
            stride = 2 if i < 4 else 1#前四层步长为2，后两层步长为1
            kernels = forward.FIRST_OUTPUT_CHANNEL / 2 * 2 ** i if i < 5 else 1#前五层通道数递增，最后一层通道数为1，用来指示这一块图是真是假
            activation_fn = forward.lrelu if i < 5 else tf.nn.sigmoid#前五层的激活函数是lrelu，最后一层用sigmoid归到0～1
            bn = forward.batchnorm if i < 5 else tf.identity#前五层进行批标准化，最后一层不做（一定不要在最后一层加批标准化）
            layers.append(activation_fn(bn(dis_conv(layers[-1], kernels, stride, i+1))))#一次通过卷积、批标准化、激活函数加入layers
        return layers[-1]#返回结果
    
    def guide_decoder(middle_layer, batch_size):#定义guide decoder，用来防止U-net底层被舍弃。与生成器中的decoder不同，他没有与encoder的skip connection
        layers = [middle_layer]#layers用来存放各层结果
        for i in range(5):#guide decoder也是六个模块儿，这个循环构建前五个
            deconvolved = forward.gen_deconv(layers[-1], forward.FIRST_OUTPUT_CHANNEL * 2 ** min(forward.MAX_OUTPUT_CHANNEL_LAYER, 4 - i), batch_size)#先是一个步长为一的反卷积
            output = forward.batchnorm(deconvolved)#通过批标准化层
            output = forward.lrelu(output)#再通过激活函数
            layers.append(output)#将结果加入layers
        output = forward.gen_deconv(output, 3, batch_size)#最后一层的反卷积
        output = tf.nn.tanh(output)#批标准化
        layers.append(output)#激活函数
        return layers[-1]#返回guide decoder的输出

    X = tf.placeholder(tf.float32, [None, None, None, 3])#输入（黑白图片）的占位符
    with tf.name_scope('generator'), tf.variable_scope('generator'):#生成器的变量名前加上generator前缀，以便与判别器的变量分开训练
        Y, middle_layer = forward.forward(X, BATCH_SIZE, True)#构建生成器网络，并获得其输出与中间层
        Y_guide = guide_decoder(middle_layer, BATCH_SIZE)#以中间层为输入构建guide decoder
    Y_real = tf.placeholder(tf.float32, [None, None, None, 3])#输出（输入对应的原彩色图片）的占位符
    XYY = tf.concat([X, Y, Y_real], axis=2)#将黑白图、生成的彩图和原彩图合并，用来展示结果
    
    with tf.name_scope('discriminator_real'):#判别真实图片的判别器的name scope
        with tf.variable_scope('discriminator'):#判别器的variable scope，为之后的变量复用作准备
            discriminator_real = discriminator(X, Y_real)#给判别器喂入黑白图及其对应的原彩图，得到一个输出

    with tf.name_scope('discriminator_fake'):#判别生成图片的判别器的name scope
        with tf.variable_scope('discriminator', reuse=True):#判别器的variable scope，复用变量
            discriminator_fake = discriminator(X, Y)#给判别器喂入黑白图及生成器生成的彩图，得到另一个输出

    dis_loss = tf.reduce_mean(-tf.log(discriminator_real + EPS) -tf.log(1 - discriminator_fake + EPS))#判别器的损失函数是两个判别器输出的交叉熵的平均
    dis_vars = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]#获得判别器的变量
    dis_optimizer = tf.train.AdamOptimizer(LEARNING_RATE, BETA1)#定义判别器的optimizer
    dis_train_op = dis_optimizer.minimize(dis_loss, var_list=dis_vars)#判别器的训练步骤，注意只训练判别器的变量

    gen_loss_GAN = tf.reduce_mean(-tf.log(discriminator_fake + EPS))#判别器提供给生成器的损失，生成器希望判别器把它生成的图片判断为原图
    gen_loss_L1 = tf.reduce_mean(tf.abs(Y - Y_real))#生成器生成的图与原图l1距离
    guide_decoder_loss =  tf.reduce_mean(tf.abs(Y_guide - Y_real))#guide decoder生成的图与原图的l1距离
    gen_loss = L1_WEIGHT * (gen_loss_L1 + GUIDE_DECODER_WEIGHT * guide_decoder_loss) + GAN_WEIGHT * gen_loss_GAN#生成器的损失函数为以上三项的加权和
    gen_vars = [var for var in tf.trainable_variables() if var.name.startswith('generator')]#获得生成器的变量
    gen_optimizer = tf.train.AdamOptimizer(LEARNING_RATE, BETA1)#定义生成器的optimizer
    gen_train_op = gen_optimizer.minimize(gen_loss, var_list=gen_vars)#生成器的训练步骤，注意只训练生成器的变量

    global_step = tf.Variable(0, trainable=False)#定义global step
    incr_global_step = tf.assign(global_step, global_step + 1) #定义global step加一的步骤

    train_op = tf.group([dis_train_op, gen_train_op, incr_global_step])#把判别器、生成器的训练步骤以及global step加一组合起来
    
    saver = tf.train.Saver()#定义用来保存、读取模型的saver
    X_batch, Y_real_batch = generateds.get_tfrecord(BATCH_SIZE, True)#从tfrecord中获取黑白图和对应彩图

    if not os.path.exists(MODEL_SAVE_PATH):#创建需要但尚未创建的模型存储目录
        os.mkdir(MODEL_SAVE_PATH)#创建需要但尚未创建的模型存储目录
    if not os.path.exists(TRAINING_RESULT_PATH):#创建需要但尚未创建的训练结果目录
        os.mkdir(TRAINING_RESULT_PATH)#创建需要但尚未创建的训练结果目录
    if not os.path.exists(GUIDE_DECODER_PATH):#创建需要但尚未创建的guide decoder效果目录
        os.mkdir(GUIDE_DECODER_PATH)#创建需要但尚未创建的guide decoder效果目录

    with tf.Session() as sess:#开启会话
        sess.run(tf.global_variables_initializer())#全局变量初始化

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)#在模型存放路径中获取模型checkpoint的状态
        if ckpt and ckpt.model_checkpoint_path:#如果存在checkpoint且可以获得其最新版本的路径
            saver.restore(sess, ckpt.model_checkpoint_path)#从模型的最新版本路径读取模型中的参数

        coord = tf.train.Coordinator()#创建一个coordinator
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)#创建读取数据的线程们

        for i in range(global_step.eval(), TOTAL_STEP):#从当前轮数到总轮数，一轮一轮训练模型
            xs, ys = sess.run([X_batch, Y_real_batch])#从tfrecord中读取x和y的下一批数据
            _, step = sess.run([train_op, global_step], feed_dict={X:xs, Y_real:ys})#执行训练步骤，并获取轮数和损失
            for i in range(4):#为了生成器和判别器的平衡，再训练四次生成器
                sess.run(gen_train_op, feed_dict={X:xs, Y_real:ys})#训练生成器
            if step % SAVE_FREQ == 0:#如果到了该保存模型的轮数
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)#保存模型
            if step % DISPLAY_FREQ == 0:#如果到了该展示训练效果的轮数
                glloss, ggloss, dloss = sess.run([gen_loss_L1, gen_loss_GAN, dis_loss], feed_dict={X:xs, Y_real:ys})#获取三部分的loss
                print('\rSteps: {}, Generator L1 loss: {:.6f}, Generator GAN loss: {:.6f}, Discriminator loss: {:.6f}'.format(step, glloss , ggloss, dloss))#输出轮数和各部分loss
                test_result = sess.run(XYY, feed_dict={X:xs, Y_real:ys})#获取黑白图、生成图、原材图的拼接
                for i, img in enumerate(test_result[:3]):#对于这批图的前三张
                    img = (img + 1) / 2#从-1～1映射到0～1
                    img *= 256#再映射到0～256
                    img = img.astype(np.uint8)#类型化为uint8
                    Image.fromarray(img).save(os.path.join(TRAINING_RESULT_PATH, 'Step{}-{}.png'.format(step, i+1)))#转成图片并保存
            if step % DISPLAY_GUIDE_DECODER_FREQ == 0:#如果到了该展示guide decoder效果的轮数
                guide_result = sess.run(Y_guide, feed_dict={X:xs, Y_real:ys})#获取guide decoder生成的图片
                for i, img in enumerate(guide_result[:1]):#对于该批图片的第一张
                    img = (img + 1) / 2#从-1～1映射到0～1
                    img *= 256#再映射到0～256
                    img = img.astype(np.uint8)#类型化为uint8
                    Image.fromarray(img).save(os.path.join(GUIDE_DECODER_PATH, 'Step-{}.png'.format(step)))#转成图片并保存
            print('\r{}'.format(step), end='')#输出训练轮数

        coord.request_stop()#要求读取图片的线程们停止
        coord.join(threads)#等待他们停止

if __name__ == '__main__':#执行此脚本时
    backward()#调用反向传播函数进行模型练

