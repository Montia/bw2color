import numpy as np#这部分引入需要的模块
import tensorflow as tf
import os
from PIL import Image
import forward
import generateds
import backward

TEST_NUM = 100#测试图片的数量
TEST_RESULT_PATH = 'test_result_l1weight={},gfc={}, mcl={} with guide encoder'.format(backward.L1_WEIGHT, forward.FIRST_OUTPUT_CHANNEL, forward.MAX_OUTPUT_CHANNEL_LAYER)#存储测试结果的目录


def test():
    X = tf.placeholder(tf.float32, [None, None, None, 3])#输入（黑白图片）的占位符
    with tf.name_scope('generator'), tf.variable_scope('generator'):#生成器的变量名前加上generator前缀，以便与判别器的变量分开训练
        Y = forward.forward(X, 1, False)#构建生成器网络，并获得其输出与中间层
    Y_real = tf.placeholder(tf.float32, [None, None, None, 3])#以中间层为输入构建guide decoder
    XYY = tf.concat([X, Y, Y_real], axis=2)#输出（输入对应的原彩色图片）的占位符

    global_step = tf.Variable(0, trainable=False)#定义global step
    saver = tf.train.Saver()#定义用来读取模型的saver

    X_batch, Y_real_batch = generateds.get_tfrecord(1, False)#从tfrecord中获取黑白图和对应彩图

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())#全局变量初始化

        ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)#在模型存放路径中获取模型checkpoint的状态
        if ckpt and ckpt.model_checkpoint_path:#如果存在checkpoint且可以获得其最新版本的路径
            saver.restore(sess, ckpt.model_checkpoint_path)#从模型的最新版本路径读取模型中的参数
        else:#没找到checkpoint话
            print('Checkpoint Not Found')#输出一下
            return#没有模型可以测试，结束运行
        
        coord = tf.train.Coordinator()#创建一个coordinator
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)#创建读取数据的线程们
        
        if not os.path.exists(TEST_RESULT_PATH):#创建需要但尚未创建的训练结果目录
            os.mkdir(TEST_RESULT_PATH)#创建需要但尚未创建的训练结果目录

        for i in range(TEST_NUM):#对于每一轮测试
            xs, ys = sess.run([X_batch, Y_real_batch])#从tfrecord中读取x和y的下一批数据（对于测试来说，一批就是一张）
            img = sess.run(XYY, feed_dict={X: xs, Y_real: ys})#获取黑白图、生成图、原材图的拼接
            img = (img + 1) / 2#从-1～1映射到0～1
            img *= 256#再映射到0～256
            img = img.astype(np.uint8)#类型化为uint8
            Image.fromarray(img[0]).save(os.path.join(TEST_RESULT_PATH, '{}.png'.format(i+1)))#转成图片并保存

        coord.request_stop()#要求读取图片的线程们停止
        coord.join(threads)#等待他们停止

if __name__ == '__main__':#执行此脚本时
    test()#调用测试函数进行测试
