from PIL import Image, ImageDraw
import numpy as np

def create_masks_in_one_image(polygons, img_h, img_w):
    """
    create mask with more polygons on one image
    :param polygons: list of polygons [x1, y1, x2, y2, ..., xn, yn]
    :param img_h: image height
    :param img_w: image width
    :return: image mask (h,w) with values {0,1} (1 where there is the object)
    """
    
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



def create_masks_multiclass(polygons_mesh, polygones_wire, img_h, img_w):
    """
    create mask with more polygons on one image
    :param polygons_mesh: list of polygons [x1, y1, x2, y2, ..., xn, yn]
    :param polygons_wire: list of polygons [x1, y1, x2, y2, ..., xn, yn]
    :param img_h: image height
    :param img_w: image width
    :return: image mask (h,w,3) with values (0,255,0) for mesh, (255,0,0) for wire and black background
    """

    return mask


def create_masks_one_hot(masks):
    """
    Create one-hot encoding masks for all masks
    :param masks: list of all the masks created with the function create_masks_multiclass
    :return: one-hot encoding masks with channel 0: background, 1: mesh, 2:wire
    """
   
    return one_hot_masks




