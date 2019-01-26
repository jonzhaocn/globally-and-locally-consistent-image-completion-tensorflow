from network_model import GloballyAndLocallyConsistentImageCompletion
from image_mask import ImageMask
import numpy as np
import os
import cv2
import random

"""
this script : use a trained model to complete images
"""

batch_size = 64
image_height = 128
image_width = 128
local_area_shape = (64, 64)
mask_side_range = (32, 64)
# whether random crop a image
whether_random_crop = True
# whether resize a image before cropping
whether_resize = False
create_central_mask = True
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def read_image_form_folder(image_folder_path, read_image_count):
    """
    get test images from a folder
    """
    image_list = os.listdir(image_folder_path)
    image_list = image_list[0:min(read_image_count, len(image_list))]
    image_paths = [os.path.join(image_folder_path, name) for name in image_list]
    images = []
    for path in image_paths[0:]:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
        img = resize_and_random_crop(img)
        if img is None:
            continue
        img = img[np.newaxis, :]
        images.append(img)
    images = np.concatenate(images, axis=0)
    return images


def resize_and_random_crop(image):
    crop_height = image_height
    crop_width = image_width

    # resize
    height, width = image.shape[:2]
    min_val = min(height, width)
    if min_val < 128:
        return None
    if whether_resize and min_val > 200:
        radio = random.randint(128, 200) / min(height, width)
        image = cv2.resize(image, (0, 0), fx=radio, fy=radio)
        height, width = image.shape[:2]
    # crop img
    if whether_random_crop:
        crop_h_s = random.randint(0, height - crop_height)
        crop_w_s = random.randint(0, width - crop_width)
    else:
        crop_h_s = (height - crop_height) // 2
        crop_w_s = (width - crop_width) // 2
    image = image[crop_h_s:crop_h_s + crop_height, crop_w_s:crop_w_s + crop_width, :]
    return image


if __name__ == '__main__':
    test_image_folder_path = '../data/img_align_celeba_for_test'
    ckpt_dir = './ckpt'
    save_result_dir = './result/'
    count = 64
    test_images = read_image_form_folder(test_image_folder_path, count)
    if not os.path.exists(ckpt_dir):
        print('ckpt dir does not exist')
    else:
        inpainting_model = GloballyAndLocallyConsistentImageCompletion(image_height, image_width, local_area_shape,
                                                                       ckpt_dir, predicting_mode=True)
        for i in range(0, test_images.shape[0], batch_size):
            i_start = i
            i_end = min(i_start+batch_size, test_images.shape[0])
            batch_image = test_images[i_start:i_end]
            image_mask_c = ImageMask(batch_image.shape[0], (image_height, image_width, 3), mask_side_range,
                                     local_area_shape, create_central_mask)
            mask = image_mask_c.get_mask_array()
            # create a masked image
            masked_images = inpainting_model.mask_image(batch_image, mask)
            # complete image
            result = inpainting_model.complete_image(masked_images, mask)
            # convert RGB to BGR for opencv python
            result = result[:, :, :, (2, 1, 0)] * 255.0
            masked_images = masked_images[:, :, :, (2, 1, 0)] * 255.0
            if not os.path.exists(save_result_dir):
                os.makedirs(save_result_dir)
            for j in range(result.shape[0]):
                completed_image = result[j, :, :, :]
                input_image = masked_images[j, :, :, :]
                cv2.imwrite(os.path.join(save_result_dir, "%d_completed.png" % (i_start+j)), completed_image)
                cv2.imwrite(os.path.join(save_result_dir, "%d_masked.png" % (i_start+j)), input_image)
