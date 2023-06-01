"""
Dataset creation for semantic segmentation model with mesh and wire together.
The mask is one hot encoded with 3 channels (mesh+wire+background)
The images and masks are cropped and reshaped.
3 arrays are created: images, masks and filenames (to keep track of the original images)
"""

import json
import numpy as np
import cv2
from mask_utils import get_xy_segmentation, create_masks_multiclass, create_masks_one_hot
import os

IMG_RES_W = 512  # output image width Potenza di 2
IMG_RES_H = 512  # output image height

src_path = "../../sources"


dst_dir_arr = "saved_arrays_multiclass"
if not os.path.exists(dst_dir_arr):
    os.makedirs(dst_dir_arr)
"""...
.
..."""

np.save(os.path.join(dst_dir_arr, "images.npy"), ...)
np.save(os.path.join(dst_dir_arr, "one_hot_masks.npy"), ...)
np.save(os.path.join(dst_dir_arr, "filenames.npy"), ...)