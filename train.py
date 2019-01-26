import tensorflow as tf
import os
from network_model import GloballyAndLocallyConsistentImageCompletion
from utils import image_dataset_iterator
from image_mask import ImageMask

"""
this script: main function for training network
"""
img_height = 128
img_width = 128
batch_size = 64
tfrecord_dir = '/data1/zhaozh/ILSVRC2012_img_train_tfrecords_128'
ckpt_dir = './ckpt'
total_image_count = 202500
local_area_shape = (64, 64)
mask_side_range = (32, 64)
create_central_mask = True

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def train():
    """
    train network
    :return:
    """
    iterator = image_dataset_iterator(tfrecord_dir, batch_size)
    image_dataset = iterator.get_next()['image']
    inpainting_model = GloballyAndLocallyConsistentImageCompletion(img_height, img_width, local_area_shape, ckpt_dir,
                                                                   batch_size=batch_size)
    sess = tf.Session()
    "get epoch start"
    epoch_start = 0
    if ckpt_dir:
        lasted_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
        if lasted_checkpoint is not None:
            epoch_start = int(lasted_checkpoint.split('/')[-1].split('-')[-1]) + 1

    iteration_of_disc = 0
    for e in range(epoch_start, 50):
        sess.run(iterator.initializer)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                images = sess.run(image_dataset)
                image_mask_c = ImageMask(batch_size, (img_height, img_width, 3), mask_side_range, local_area_shape, create_central_mask)
                image_mask_d = ImageMask(batch_size, (img_height, img_width, 3), mask_side_range, local_area_shape, create_central_mask)
                mask_c = image_mask_c.get_mask_array()
                local_area_top_left_c = image_mask_c.local_area_top_left
                local_area_top_left_d = image_mask_d.local_area_top_left
                if images.shape[0] != batch_size:
                    print('count of ground true is not equal to batch size')
                    break
                if e < 9:
                    "-----training completion network"
                    inpainting_model.train_completion_network(images, mask_c)
                elif e < 10 and iteration_of_disc < 1000:
                    "-----training discriminator"
                    iteration_of_disc = iteration_of_disc+1
                    inpainting_model.train_discriminator(images, mask_c, local_area_top_left_c, local_area_top_left_d)
                else:
                    "-----training completion network and discriminator"
                    inpainting_model.train_completion_network_and_discriminator_jointly(images, mask_c, local_area_top_left_c, local_area_top_left_d)
        except tf.errors.OutOfRangeError:
            print('Done training, epoch reached')
        finally:
            coord.request_stop()
            coord.join(threads)
        "save model in the end of epoch"
        inpainting_model.save_model(e)


if __name__ == '__main__':
    train()