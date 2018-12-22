from forward import forward
import tensorflow as tf
import numpy as np
import generateds
import os
from PIL import Image

MODEL_SAVE_PATH = 'unet/model'
TRAINING_RESULT_PATH = 'unet/result'


def backward():
    X = tf.placeholder(tf.float32, [None, 512, 512, 1])
    Y = tf.placeholder(tf.float32, [None, 512, 512, 1])

    y_ = forward(X, 1, True)

    # loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=y_)
    loss = tf.losses.sigmoid_cross_entropy(Y, y_)

    # loss_mean = tf.reduce_mean(loss)
    # tf.add_to_collection(name='loss', value=loss_mean)
    # loss_all = tf.add_n(inputs=tf.get_collection(key='loss'))
    # train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_all)

    # correct_prediction = tf.equal(tf.argmax(input=y_, axis=3, output_type=tf.int32), Y)
    # correct_prediction = tf.cast(correct_prediction, tf.float32)
    # accuracy = tf.reduce_mean(correct_prediction)

    global_step = tf.Variable(0, trainable=False)
    incr_global_step = tf.assign(global_step, global_step + 1)
    train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
    train_op = tf.group([train_step, incr_global_step])

    saver = tf.train.Saver()
    X_batch, Y_real_batch = generateds.get_tfrecord(1, True)

    if not os.path.exists(MODEL_SAVE_PATH):
        os.mkdir(MODEL_SAVE_PATH)
    if not os.path.exists(TRAINING_RESULT_PATH):
        os.mkdir(TRAINING_RESULT_PATH)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(global_step.eval(), 10000):
            xs, ys = sess.run([X_batch, Y_real_batch])
            _, step = sess.run([train_op, global_step], feed_dict={X: xs, Y: ys})

            print(i)
            if step % 50 == 0:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, 'unet_model'), global_step=global_step)
                los = sess.run(loss, feed_dict={X:xs, Y: ys})
                print('after %d step training, loss is: %f' % (step, los))
                test_result = sess.run(y_, feed_dict={X: xs})
                img = test_result
                img = (img + 1) / 2
                img *= 256
                img = img.astype(np.uint8)
                img = img.reshape(512, 512)
                Image.fromarray(img, 'L').save(TRAINING_RESULT_PATH + '/' + str(step) + '.tif')

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    backward()
