import numpy as np
from PIL import Image
import tensorflow as tf
import os
from tqdm import tqdm

image_train_path = './U-net/data_set/train'
image_train_label = './U-net/data_set/label'
image_test_path = './U-net/data_set/test'
tfRecord_train = './tfrecord/unet_train.tfrecords'
tfRecord_test = './tfrecord/unet_test.tfrecords'
data_path = './tfrecord'
image_shape = [1, 512, 512, 1]


def write_tfRecord(tfRecordName, image_path, label_path):
    writer = tf.python_io.TFRecordWriter(tfRecordName)

    for img_file in tqdm(os.listdir(image_path)):
        img = Image.open(image_path + '/' + img_file)
        label = Image.open(label_path + '/' + img_file)

        example = tf.train.Example(features=tf.train.Features(feature={
            'X': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
            'Y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tobytes()]))
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
    write_tfRecord(tfRecord_train, image_train_path, image_train_label)
    # write_tfRecord(tfRecord_test, image_test_path)


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
    X = tf.cast(X, tf.float32) / 128 - 1

    Y = tf.decode_raw(features['Y'], tf.uint8)
    Y = tf.reshape(Y, image_shape)
    Y = tf.cast(Y, tf.float32) / 128 - 1

    return X, Y


def get_tfrecord(num, isTrain=True):
    image_shape[0] = num
    tfRecord_path = tfRecord_train if isTrain else tfRecord_test

    X, Y = read_tfRecord(tfRecord_path)
    return X, Y


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
            arr = arr.reshape([512,512])
            print(arr.shape)
            print(arr.dtype)
            img = Image.fromarray(arr, 'L')
            img.save('%d.tif' % i)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    generate_tfRecord()
    # test_get_tfrecord()
