import json
import numpy as np
import cv2
from multiclass_mask_utils import  mask_one_hot
import os

SAVE_MASKS = True # if you want to save mask images for a visual check



src_path = "mesh_and_wire/masks/"



dst_dir_arr = "selo/arr/"
if not os.path.exists(dst_dir_arr):
    os.mkdir(dst_dir_arr)



if SAVE_MASKS:
    dst_dir_masks = "selo/masks/"
    if not os.path.exists(dst_dir_masks):
        os.makedirs(dst_dir_masks)




masks = np.load('multiclass_saved_arrays/_multiclass_masks.npy')
mask_one_hot(masks)
OHE_masks = mask_one_hot(masks)

print("masks", masks.shape)
print("OHE_masks", OHE_masks.shape)

#np.save(dst_dir_arr + "selo_multiclass_masks.npy", OHE_masks)
