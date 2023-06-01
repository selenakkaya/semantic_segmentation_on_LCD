
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
    
  
    return mask

#------------------------------------------------------------------------------------------#
#----------------------------------One Hot Encoding----------------------------------------#
#------------------------------------------------------------------------------------------#


filenames = np.load('multiclass_saved_arrays\_multiclass_filenames.npy')
def mask_one_hot(masks):
   
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
  
    return mask


def get_xy_segmentation(segment):
    """
    Transform polygon into [x1, y1, x2, y2, ..., xn, yn] format
    :param segment: segmentation in labelme format
    :return: polygon [x1, y1, x2, y2, ..., xn, yn]
    """
  
    return polygon
