import numpy as np
from PIL import Image
import tensorflow as tf 
from time import sleep
import os
import forward
import generateds

BATCH_SIZE = 1
L1_WEIGHT = 100
GAN_WEIGHT = 1
EPS = 1e-12
LEARNING_RATE = 2e-04
BETA1 = 0.5
EMA_DECAY = 0.98
MODEL_SAVE_PATH = './model'
MODEL_NAME = 'pix2pix_model'
TOTAL_STEP = 100000
TRAINING_RESULT_PATH = 'training_result'

def backward():
    def dis_conv(X, kernels, stride, layer, regularizer=None):
        initializer = tf.truncated_normal_initializer(0, 0.2)
        w = tf.get_variable('w{}'.format(layer), [forward.KERNEL_SIZE, forward.KERNEL_SIZE, X.get_shape().as_list()[-1], kernels], initializer=initializer)
        padded_X = tf.pad(X, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
        #print(w)
        return tf.nn.conv2d(padded_X, w, [1, stride, stride, 1], padding='VALID')

    def discriminator(discriminator_input, discriminator_output):
        X = tf.concat([discriminator_input, discriminator_output], axis=3)
        layers = [X]
        for i in range(5):
            stride = 2 if i < 3 else 1
            kernels = forward.FIRST_OUTPUT_CHANNEL * 2 ** i if i < 4 else 1
            activation_fn = forward.lrelu if i < 4 else tf.nn.sigmoid
            layers.append(activation_fn(forward.batchnorm(dis_conv(layers[-1], kernels, stride, i+1))))
        #for layer in layers:
        #    print(layer)
        return layers[-1]

    X = tf.placeholder(tf.float32, [None, 256, 256, 3])
    with tf.name_scope('generator'), tf.variable_scope('generator'):
        Y = forward.forward(X, BATCH_SIZE, True)
    Y_real = tf.placeholder(tf.float32, [None, 256, 256, 3])
    XYY = tf.concat([X, Y, Y_real], axis=2)

    with tf.name_scope('discriminator_real'):
        with tf.variable_scope('discriminator'):
            discriminator_real = discriminator(X, Y_real)

    with tf.name_scope('discriminator_fake'):
        with tf.variable_scope('discriminator', reuse=True):
            discriminator_fake = discriminator(X, Y)
    
    gen_loss_GAN = tf.reduce_mean(-tf.log(discriminator_fake + EPS))
    gen_loss_L1 = tf.reduce_mean(tf.abs(Y - Y_real))
    gen_loss = L1_WEIGHT * gen_loss_L1 + GAN_WEIGHT * gen_loss_GAN
    gen_vars = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
    gen_optimizer = tf.train.AdamOptimizer(LEARNING_RATE, BETA1)
    gen_grads_and_vars = gen_optimizer.compute_gradients(gen_loss, var_list=gen_vars)
    gen_training_op = gen_optimizer.apply_gradients(gen_grads_and_vars)

    dis_loss = tf.reduce_mean(-tf.log(discriminator_real + EPS) -tf.log(1 - discriminator_fake + EPS))
    dis_vars = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
    dis_optimizer = tf.train.AdadeltaOptimizer(LEARNING_RATE, BETA1)
    dis_grads_and_vars = dis_optimizer.compute_gradients(dis_loss, var_list=dis_vars)
    dis_training_op = dis_optimizer.apply_gradients(dis_grads_and_vars)
    
    ema = tf.train.ExponentialMovingAverage(EMA_DECAY)
    ema_op = ema.apply(tf.trainable_variables())

    global_step = tf.Variable(0, trainable=False)
    incr_global_step = tf.assign(global_step, global_step + 1)
    
    train_op = tf.group([gen_training_op, dis_training_op, ema_op, incr_global_step])

    saver = tf.train.Saver()
    X_batch, Y_real_batch = generateds.get_tfrecord(BATCH_SIZE, True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        for i in range(global_step.eval(), TOTAL_STEP):
            xs, ys = sess.run([X_batch, Y_real_batch])
            _, step = sess.run([train_op, global_step], feed_dict={X:xs, Y_real:ys})
            if step % 500 == 0:
                gloss, dloss = sess.run([gen_loss, dis_loss], feed_dict={X:xs, Y_real:ys})
                print('\rAfter {} steps, the loss of generator is {}, the loss of discriminator is {}'.format(step, gloss, dloss))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                test_result = sess.run(XYY, feed_dict={X:xs, Y_real:ys})
                if not os.path.exists(TRAINING_RESULT_PATH):
                    os.mkdir(TRAINING_RESULT_PATH)
                for i, img in enumerate(test_result):
                    img = (img + 1) / 2
                    img *= 256
                    img = img.astype(np.uint8)
                    Image.fromarray(img).save(os.path.join(TRAINING_RESULT_PATH, 'Step-{}.png'.format(step)))
            print('\r{}'.format(step), end='')


if __name__ == '__main__':
    backward()

