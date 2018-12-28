import numpy as np
from PIL import Image
import tensorflow as tf
import os
from tqdm import tqdm

image_train_path = './out/crop/train'
image_test_path = './out/crop/test'
tfRecord_train = './tfrecord/pix2pix_train.tfrecords'
tfRecord_test = './tfrecord/pix2pix_test.tfrecords'
data_path = './tfrecord'
image_shape = [512, 512, 3]


def write_tfRecord(tfRecordName, image_path):
    writer = tf.python_io.TFRecordWriter(tfRecordName)

    for img_file in tqdm(os.listdir(image_path)):
        path = image_path + '/' + img_file
        img = Image.open(path)

        color_img = img.crop((0, 0, img.size[0] // 2, img.size[1]))
        grey_img = img.crop((img.size[0] // 2, 0, img.size[0], img.size[1]))
        X_shape = list(grey_img.size)
        X_shape.append(3)
        Y_shape = list(color_img.size)
        Y_shape.append(3)
        assert(X_shape == Y_shape)
        example = tf.train.Example(features=tf.train.Features(feature={
            'X_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=X_shape)),
            'Y_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=Y_shape)),
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
                                           'X_shape': tf.FixedLenFeature([3], tf.int64),
                                           'Y_shape': tf.FixedLenFeature([3], tf.int64),
                                           'X': tf.FixedLenFeature([], tf.string),
                                           'Y': tf.FixedLenFeature([], tf.string)
                                       })
    X = tf.decode_raw(features['X'], tf.uint8)
    X = tf.reshape(X, features['X_shape'])
    # X = tf.cast(X, tf.float32) * (1. / 255)
    X = tf.cast(X, tf.float32) / 128 - 1
    Y = tf.decode_raw(features['Y'], tf.uint8)
    Y = tf.reshape(Y, features['Y_shape'])
    # Y = tf.cast(Y, tf.float32) * (1. / 255)
    Y = tf.cast(Y, tf.float32) / 128 - 1

    return X, Y


def get_tfrecord(num, isTrain=True):
    #image_shape[0] = num
    tfRecord_path = tfRecord_train if isTrain else tfRecord_test

    img_batch, label_batch = [], []
    for i in range(num):
        X, Y = read_tfRecord(tfRecord_path)
        img_batch.append(X)
        label_batch.append(Y)
    #img_batch, label_batch = tf.train.shuffle_batch([X, Y], batch_size=num, num_threads=2, capacity=100, min_after_dequeue=50)
    return img_batch, label_batch
    #return X, Y


# rebuild image to check
def test_get_tfrecord():
    x, y = get_tfrecord(1, True)
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in [1, 2, 3]:
            xs, ys = sess.run([x, y])
            arr = ((ys + 1) * 128).astype(np.uint8)
            img = Image.fromarray(arr)
            img.save('%d.jpg' % i)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    generate_tfRecord()
    # test_get_tfrecord()
