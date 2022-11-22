from PIL import Image, ImageDraw
import numpy as np
import cv2


def create_masks_in_one_image(polygons, img_h, img_w):
    """
    create mask with more polygons on one image
    :param polygons: list of polygons [x1, y1, x2, y2, ..., xn, yn]
    :param img_h: image height
    :param img_w: image width
    :return: image mask (h,w) with values {0,1} (1 where there is the object)
    """
    black_img = Image.new('L', (img_w, img_h), 0)
    for polygon in polygons:
        ImageDraw.Draw(black_img).polygon(polygon, outline=1, fill=1)
    mask = np.array(black_img)
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
    ImageDraw.Draw(black_img).polygon(polygon, outline=1, fill=1)
    mask = np.array(black_img)
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




