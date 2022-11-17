import json
import numpy as np
import cv2
from multiclass_mask_utils import get_xy_segmentation, create_masks_in_one_image
import os

IMG_RES_W = 512  # output image width Potenza di 2
IMG_RES_H = 512  # output image height
SAVE_IMAGES = True # if you want to save mask images for a visual check
CATEGORY_M = "mesh"
CATEGORY_W = "wire" 
src_path = "sources"



def draw_image_and_mask(img_res_h, img_res_w, img_res, mask_res):
    """
    Draw the image on the left and the relative mask on the right
    :param img_res_h: image height
    :param img_res_w: image width
    :param img_res: image cropped and resized
    :param mask_res: mask cropped and resized
    :return: image + mask in a single image
    """
    double_img = np.zeros((img_res_h, img_res_w * 2, 3))
    img_res = cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR) # transform to BGR to save image with opencv
    double_img[:, :img_res_w, :] = img_res
    double_img[:, img_res_w:, :] = mask_res
    return double_img

dst_dir_arr = "multiclass_saved_arrays"
if not os.path.exists(dst_dir_arr):
    os.mkdir(dst_dir_arr)


if SAVE_IMAGES:
    dst_dir_imgs = "mesh_and_wire/mask_and_images"
   
    if not os.path.exists(dst_dir_imgs):
        os.makedirs(dst_dir_imgs)


images = []
masks = []
filenames = []

for site in os.listdir(src_path):
    name_list = [x.split(".")[0] for x in os.listdir(os.path.join(src_path, site, "images"))]
    print(name_list)
    for filename in name_list:
        ann_path = src_path + '/' + site + "/labels/" + filename + ".json"
        img = cv2.imread(src_path + '/' + site + "/images/" + filename + ".JPG")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #opencv open images in BGR
        img_w = img.shape[1]
        img_h = img.shape[0]
        l = abs(img_w - img_h) // 2

        # get polygons from json file
        with open(ann_path, 'r') as f:
            dataset = json.loads(f.read())
            annots = dataset["shapes"]
            elements_mesh = [x for x in annots if x['label'] == CATEGORY_M]
            elements_wire = [x for x in annots if x['label'] == CATEGORY_W]

        polygones_wire = []
        polygones_mesh = []
        for element_m in elements_mesh:
            s_1 = element_m["points"]
            polygon_m = get_xy_segmentation(s_1)
            polygones_mesh.append(polygon_m)
        
        for element_w in elements_wire:
            s_2 = element_w["points"]
            polygon_w = get_xy_segmentation(s_2)
            polygones_wire.append(polygon_w)

        mask = create_masks_in_one_image(polygones_mesh, polygones_wire, img.shape[0], img.shape[1])

        img_arr = np.array(img)
        if img_w>img_h:
            # center
            cut_mask_cx = mask[:,l:-l]
            mask_res_cx = cv2.resize(cut_mask_cx, (IMG_RES_W, IMG_RES_H))
            if mask_res_cx.any(): #avoid images without mesh
                cut_img_cx = img_arr[:, l:-l, :]
                img_res_cx = cv2.resize(cut_img_cx, (IMG_RES_W, IMG_RES_H))
                images.append(img_res_cx)
                masks.append(mask_res_cx)
                filenames.append(filename + '_cx')
                if SAVE_IMAGES:
                    double_img_cx = draw_image_and_mask(IMG_RES_H, IMG_RES_W, img_res_cx, mask_res_cx)
                    cv2.imwrite(dst_dir_imgs + '/' + filename + '_cx.jpg', double_img_cx)

            # left
            cut_mask_sx = mask[:, :img_h]
            mask_res_sx = cv2.resize(cut_mask_sx, (IMG_RES_W, IMG_RES_H))
            if mask_res_sx.any():
                cut_img_sx = img_arr[:, :img_h, :]
                img_res_sx = cv2.resize(cut_img_sx, (IMG_RES_W, IMG_RES_H))
                images.append(img_res_sx)
                masks.append(mask_res_sx)
                filenames.append(filename + '_sx')
                if SAVE_IMAGES:
                    double_img_sx = draw_image_and_mask(IMG_RES_H, IMG_RES_W, img_res_sx, mask_res_sx)
                    cv2.imwrite(dst_dir_imgs + '/' + filename + '_sx.jpg', double_img_sx)

            # right
            cut_mask_dx = mask[:, (img_w - img_h):]
            mask_res_dx = cv2.resize(cut_mask_dx, (IMG_RES_W, IMG_RES_H))
            if mask_res_dx.any():
                cut_img_dx = img_arr[:, (img_w - img_h):, :]
                img_res_dx = cv2.resize(cut_img_dx, (IMG_RES_W, IMG_RES_H))
                images.append(img_res_dx)
                masks.append(mask_res_dx)
                filenames.append(filename + '_dx')
                if SAVE_IMAGES:
                    double_img_dx = draw_image_and_mask(IMG_RES_H, IMG_RES_W, img_res_dx, mask_res_dx)
                    cv2.imwrite(dst_dir_imgs + '/' + filename + '_dx.jpg', double_img_dx)

        # vertical images
        else:
            cut_mask_cy = mask[l:-l, :]
            mask_res_cy = cv2.resize(cut_mask_cy, (IMG_RES_W, IMG_RES_H))
            if mask_res_cy.any():
                cut_img_cy = img_arr[l:-l, :, :]
                img_res_cy = cv2.resize(cut_img_cy, (IMG_RES_W, IMG_RES_H))
                images.append(img_res_cy)
                masks.append(mask_res_cy)
                filenames.append(filename + '_cy')
                if SAVE_IMAGES:
                    double_img_cy = draw_image_and_mask(IMG_RES_H, IMG_RES_W, img_res_cy, mask_res_cy)
                    cv2.imwrite(dst_dir_imgs + '/' + filename + '_cy.jpg', double_img_cy)

            cut_mask_by = mask[(img_h - img_w):, :]
            mask_res_by = cv2.resize(cut_mask_by, (IMG_RES_W, IMG_RES_H))
            if mask_res_by.any():
                cut_img_by = img_arr[(img_h - img_w):, :, :]
                img_res_by = cv2.resize(cut_img_by, (IMG_RES_W, IMG_RES_H))
                images.append(img_res_by)
                masks.append(mask_res_by)
                filenames.append(filename + '_by')
                if SAVE_IMAGES:
                    double_img_by = draw_image_and_mask(IMG_RES_H, IMG_RES_W, img_res_by, mask_res_by)
                    cv2.imwrite(dst_dir_imgs + '/' + filename + '_by.jpg', double_img_by)


