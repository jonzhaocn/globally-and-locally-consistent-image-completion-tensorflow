import random
import numpy as np

"""
image mask class: create masks to mask images
"""


class ImageMask:
    def __init__(self, batch_size, image_shape, mask_side_range=(32, 64), local_area_shape=(64, 64), create_central_mask=True):
        self._batch_size = batch_size
        self._img_height = image_shape[0]
        self._img_width = image_shape[1]
        # length range of the mask
        self._mask_side_range = mask_side_range
        self._local_area_shape = local_area_shape
        if create_central_mask:
            self._mask_array = self._create_central_mask()
        else:
            self._mask_array = self._create_random_rect_mask()

    def _create_random_rect_mask(self):
        # ---local area top left point---
        local_area_h_s = random.randint(0, self._img_height - self._local_area_shape[0])
        local_area_w_s = random.randint(0, self._img_width - self._local_area_shape[1])
        self.local_area_top_left = (local_area_h_s, local_area_w_s)
        # ---missing area height and width
        missing_area_h = random.randint(self._mask_side_range[0], self._mask_side_range[1])
        missing_area_w = random.randint(self._mask_side_range[0], self._mask_side_range[1])
        # missing area height start, width start
        missing_area_h_s = self.local_area_top_left[0] + random.randint(0, self._local_area_shape[0]-missing_area_h)
        missing_area_w_s = self.local_area_top_left[1] + random.randint(0, self._local_area_shape[1]-missing_area_w)
        # create mask
        mask = np.ones((self._batch_size, self._img_height, self._img_width, 3), dtype=np.float32)
        mask[:, missing_area_h_s:missing_area_h_s+missing_area_h, missing_area_w_s:missing_area_w_s+missing_area_w, :]=0
        return mask

    def _create_central_mask(self):
        # ---local area top left point
        local_area_h_s = (self._img_height - self._local_area_shape[0])//2
        local_area_w_s = (self._img_width - self._local_area_shape[1])//2
        self.local_area_top_left = (local_area_h_s, local_area_w_s)
        # use the max mask side length as the mask height
        missing_area_h = self._mask_side_range[1]
        missing_area_w = self._mask_side_range[1]
        missing_area_h_s = self.local_area_top_left[0] + (self._local_area_shape[0] - missing_area_h)//2
        missing_area_w_s = self.local_area_top_left[1] + (self._local_area_shape[1] - missing_area_w)//2
        # create mask
        mask = np.ones((self._batch_size, self._img_height, self._img_width, 3), dtype=np.float32)
        mask[:, missing_area_h_s:missing_area_h_s + missing_area_h, missing_area_w_s:missing_area_w_s + missing_area_w, :] = 0
        return mask

    def get_mask_array(self):
        return self._mask_array
