import numpy as np
from PIL import Image
import tensorflow as tf
import os
from tqdm import tqdm


def write_tfRecord(tfRecordName, image_path):
    writer = tf.python_io.TFRecordWriter(tfRecordName)

    for img_file in tqdm(os.listdir(image_path)):
        img = Image.open(image_path + '/' + img_file)

        color_img = img.crop((0, 0, 256, 256))
        grey_img = img.crop((256, 0, 512, 256))

        example = tf.train.Example(features=tf.train.Features(feature={
            'X': tf.train.Feature(bytes_list=tf.train.BytesList(value=[grey_img.tobytes()])),
            'Y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[color_img.tobytes()]))
        }))
        writer.write(example.SerializeToString())

    writer.close()
    print("write tfrecord successful")


def generate_tfRecord():
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print("create data dir")
    else:
        print('data dir already exists')
    write_tfRecord(tfRecord_train, image_train_path)
    write_tfRecord(tfRecord_test, image_test_path)


def read_tfRecord(tfRecord_path):
    filename_queue = tf.train.string_input_producer([tfRecord_path])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'X': tf.FixedLenFeature([], tf.string),
                                           'Y': tf.FixedLenFeature([], tf.string)
                                       })
    X = tf.decode_raw(features['X'], tf.uint8)
    X = tf.reshape(X, image_shape)
    X = tf.cast(X, tf.float32) * (1. / 255)
    Y = tf.decode_raw(features['Y'], tf.uint8)
    Y = tf.reshape(Y, image_shape)
    Y = tf.cast(Y, tf.float32) * (1. / 255)

    return X, Y


def get_tfrecord(num, isTrain=True):
    tfRecord_path = tfRecord_train if isTrain else tfRecord_test

    X, Y = read_tfRecord(tfRecord_path)
    # img_batch, label_batch = tf.train.shuffle_batch([X, Y], batch_size=num, num_threads=2, capacity=50000, min_after_dequeue=49880)
    # return img_batch, label_batch
    return X, Y


# rebuild image to check
def test_get_tfrecord():
    x, y = get_tfrecord(1)
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in [1, 2, 3]:
            xs, ys = sess.run([x, y])
            arr = ((xs + 1) * 128).astype(np.uint8)
            img = Image.fromarray(arr)
            img.save('%d.jpg' % i)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    image_train_path = './out/test/train'
    image_test_path = './out/test/test'
    tfRecord_train = './tfrecord/pix2pix_train.tfrecords'
    tfRecord_test = './tfrecord/pix2pix_test.tfrecords'
    data_path = './tfrecord'
    image_shape = [256, 256, 3]

    # test_get_tfrecord()
# generate_tfRecord()
