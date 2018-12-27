import numpy as np
import tensorflow as tf
import os
from PIL import Image
import forward
import generateds
import backward

TEST_NUM = 200
TEST_RESULT_PATH = 'test_result_l1weight={},gfc={}, mcl={}'.format(backward.L1_WEIGHT, forward.FIRST_OUTPUT_CHANNEL, forward.MAX_OUTPUT_CHANNEL_LAYER)


def test():
    X = tf.placeholder(tf.float32, [None, 512, 512, 3])
    with tf.name_scope('generator'), tf.variable_scope('generator'):
        Y = forward.forward(X, 1, False)
    Y_real = tf.placeholder(tf.float32, [None, 512, 512, 3])
    XYY = tf.concat([X, Y, Y_real], axis=2)

    #ema = tf.train.ExponentialMovingAverage(backward.EMA_DECAY)
    global_step = tf.Variable(0, trainable=False)
    #saver = tf.train.Saver(ema.variables_to_restore())
    saver = tf.train.Saver()

    X_batch, Y_real_batch = generateds.get_tfrecord(1, False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Checkpoint Not Found')
            return
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        if not os.path.exists(TEST_RESULT_PATH):
            os.mkdir(TEST_RESULT_PATH)

        for i in range(TEST_NUM):
            xs, ys = sess.run([X_batch, Y_real_batch])
            img = sess.run(XYY, feed_dict={X: xs, Y_real: ys})
            img = (img + 1) / 2
            img *= 256
            img = img.astype(np.uint8)
            Image.fromarray(img[0]).save(os.path.join(TEST_RESULT_PATH, '{}.png'.format(i+1)))

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    test()
