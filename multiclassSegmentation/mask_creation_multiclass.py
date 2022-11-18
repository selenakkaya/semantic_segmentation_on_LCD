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

images = []
masks = []
filenames = []
for site in os.listdir(src_path):
    name_list = [x.split(".")[0] for x in os.listdir(os.path.join(src_path, site, "labels"))]
    print(name_list)
    for filename in name_list:
        ann_path = src_path + '/' + site + "/labels/" + filename + ".json"
        img = cv2.imread(src_path + '/' + site + "/images/" + filename + ".jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #opencv open images in BGR
        img_w = img.shape[1]
        img_h = img.shape[0]
        l = abs(img_w - img_h) // 2

        # get polygons from json file
        with open(ann_path, 'r') as f:
            dataset = json.loads(f.read())
            annots = dataset["shapes"]
            meshes = [x for x in annots if x['label'] == "mesh"]
            wires = [x for x in annots if x['label'] == "wire"]

        polygons_mesh = []
        for mesh in meshes:
            s = mesh["points"]
            polygon = get_xy_segmentation(s)
            polygons_mesh.append(polygon)

        polygons_wire = []
        for wire in wires:
            s = wire["points"]
            polygon = get_xy_segmentation(s)
            polygons_wire.append(polygon)

        mask = create_masks_multiclass(polygons_mesh, polygons_wire, img.shape[0], img.shape[1])

        img_arr = np.array(img)
        if img_w > img_h:
            # center
            cut_img_cx = img_arr[:, l:-l, :]
            img_res_cx = cv2.resize(cut_img_cx, (IMG_RES_W, IMG_RES_H))
            images.append(img_res_cx)
            cut_mask_cx = mask[:, l:-l, :]
            mask_res_cx = cv2.resize(cut_mask_cx, (IMG_RES_W, IMG_RES_H))
            masks.append(mask_res_cx)
            filenames.append(filename + '_cx')

            # left
            cut_img_sx = img_arr[:, :img_h, :]
            img_res_sx = cv2.resize(cut_img_sx, (IMG_RES_W, IMG_RES_H))
            images.append(img_res_sx)
            cut_mask_sx = mask[:, :img_h,:]
            mask_res_sx = cv2.resize(cut_mask_sx, (IMG_RES_W, IMG_RES_H))
            masks.append(mask_res_sx)
            filenames.append(filename + '_sx')

            # right
            cut_img_dx = img_arr[:, (img_w - img_h):, :]
            img_res_dx = cv2.resize(cut_img_dx, (IMG_RES_W, IMG_RES_H))
            images.append(img_res_dx)
            cut_mask_dx = mask[:, (img_w - img_h):,:]
            mask_res_dx = cv2.resize(cut_mask_dx, (IMG_RES_W, IMG_RES_H))
            masks.append(mask_res_dx)
            filenames.append(filename + '_dx')

        # vertical images
        else:
            # center
            cut_img_cy = img_arr[l:-l, :, :]
            img_res_cy = cv2.resize(cut_img_cy, (IMG_RES_W, IMG_RES_H))
            images.append(img_res_cy)
            cut_mask_cy = mask[l:-l, :,:]
            mask_res_cy = cv2.resize(cut_mask_cy, (IMG_RES_W, IMG_RES_H))
            masks.append(mask_res_cy)
            filenames.append(filename + '_cy')

            # bottom
            cut_img_by = img_arr[(img_h - img_w):, :, :]
            img_res_by = cv2.resize(cut_img_by, (IMG_RES_W, IMG_RES_H))
            images.append(img_res_by)
            cut_mask_by = mask[(img_h - img_w):, :,:]
            mask_res_by = cv2.resize(cut_mask_by, (IMG_RES_W, IMG_RES_H))
            masks.append(mask_res_by)
            filenames.append(filename + '_by')

images = np.array(images)
filenames = np.array(filenames)
one_hot_masks = create_masks_one_hot(masks)

print("images", images.shape)
print("masks one hot", one_hot_masks.shape)

np.save(os.path.join(dst_dir_arr, "images.npy"), images)
np.save(os.path.join(dst_dir_arr, "one_hot_masks.npy"), one_hot_masks)
np.save(os.path.join(dst_dir_arr, "filenames.npy"), filenames)