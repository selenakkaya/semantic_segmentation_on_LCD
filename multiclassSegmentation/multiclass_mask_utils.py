
from PIL import Image, ImageDraw
import numpy as np
import cv2
import tensorflow as tf


def create_masks_in_one_image(polygons_mesh, polygones_wire, img_h, img_w):
    """
    create mask with more polygons on one image
    :param polygons: list of polygons [x1, y1, x2, y2, ..., xn, yn]
    :param img_h: image height
    :param img_w: image width
    :return: image mask (h,w) with values {0,1} (1 where there is the object)
    """
    
    black_img = Image.new('RGB', (img_w, img_h), (0, 0, 0))
    for p_m in polygons_mesh:
        print(p_m)
        ImageDraw.Draw(black_img).polygon(p_m, outline=(0, 255, 0), fill=(0, 255, 0))
    for p_w in polygones_wire:  
        ImageDraw.Draw(black_img).polygon(p_w, outline=(255, 0 ,0), fill=(255,0,0))
    mask = np.array(black_img)
    return mask

#------------------------------------------------------------------------------------------#
#----------------------------------One Hot Encoding----------------------------------------#
#------------------------------------------------------------------------------------------#


filenames = np.load('multiclass_saved_arrays\_multiclass_filenames.npy')
def mask_one_hot(masks):
    labels_old = [[0, 0, 0], [0, 255, 0], [255, 0, 0]]
    labels_new = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    dst_dir_arr = "selo/arr/"
    dst_dir_masks = "selo/masks/"
    for mask in masks:
        for i, label in enumerate(labels_old):
            mask[np.all(mask == label, axis = -1)] = labels_new[i]
        np.save(dst_dir_arr + "selo_multiclass_masks.npy", mask)
        for fn in filenames:
            cv2.imwrite(dst_dir_masks + fn +'_selo_by.jpg', mask)

    return mask


def create_mask(polygon, img_h, img_w):
    """
    From list with points of the polygon return the image with mask image.
    The dimension of the mask is the same of the relative image
    :param polygon: [x1, y1, x2, y2, ..., xn, yn]
    :param img_h: image height
    :param img_w: image width
    :return: image mask (h,w) with values {0,1}
    """
    black_img = Image.new('L', (img_w, img_h), 0)
    img = ImageDraw.Draw(black_img).polygon(polygon, outline=1, fill=1)
    img = ImageDraw.Draw(img).polygon(polygon, outline=2, fill=2)
    mask = np.array(img)
    return mask


def get_xy_segmentation(segment):
    """
    Transform polygon into [x1, y1, x2, y2, ..., xn, yn] format
    :param segment: segmentation in labelme format
    :return: polygon [x1, y1, x2, y2, ..., xn, yn]
    """
    polygon = []
    for p in segment:
        polygon.append(p[0])
        polygon.append(p[1])

    return polygon
