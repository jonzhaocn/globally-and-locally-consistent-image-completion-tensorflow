import os
import tensorflow as tf
import numpy as np
import cv2

"""
reference:
https://github.com/YJango/TFRecord-Dataset-Estimator-API/blob/master/TensorFlow%20Dataset%20%2B%20TFRecords.ipynb
https://stackoverflow.com/questions/44549245/how-to-use-tensorflow-tf-train-string-input-producer-to-produce-several-epochs-d/44551409#44551409
"""
img_height = 128
img_width = 128
img_channel = 3


def image_dataset_iterator(tfrecord_dir, batch_size):
    """
    read image from tfrecords file
    :param tfrecord_dir:
    :param batch_size:
    :return:
    """
    tfrecord_list = os.listdir(tfrecord_dir)
    tfrecord_names = [os.path.join(tfrecord_dir, name) for name in tfrecord_list]
    dataset = tf.data.TFRecordDataset(tfrecord_names)
    dataset = dataset.map(parse_function).shuffle(buffer_size=10000).batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    return iterator


def parse_function(example_proto):
    """
    parse function for saving images into tfrecords file
    :param example_proto:
    :return:
    """
    dics = {'image': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'image_shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64)}
    parsed_example = tf.parse_single_example(example_proto, dics)
    parsed_example['image'] = tf.decode_raw(parsed_example['image'], tf.uint8)
    parsed_example['image'] = tf.reshape(parsed_example['image'], (img_height, img_width, img_channel))
    parsed_example['image'] = tf.cast(parsed_example['image'], tf.float32)/256.0
    return parsed_example


def save_sample(images, size, path):
    """
    concat a list images to a image, and save the image
    :param images:
    :param size:
    :param path:
    :return:
    """
    pardir, _ = os.path.split(path)
    if not os.path.exists(pardir):
        os.mkdir(pardir)

    h, w = images.shape[1], images.shape[2]

    # create a large array for storing images
    merge_img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        if j >= size[0]:
            break
        merge_img[j * h:j * h + h, i * w:i * w + w, :] = image

    merge_img = merge_img*255.0
    # convert RGB to BGR for cv2.imwrite
    merge_img = merge_img[:, :, (2, 1, 0)]
    cv2.imwrite(path, merge_img)