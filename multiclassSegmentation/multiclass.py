import random
import numpy as np
#from keras.utils import normalize
import os
import glob
import cv2
from matplotlib import pyplot as plt

random.seed(1)

CATEGORY = "multiclass" #mesh and wire
SIZE = 512
N_CLASSES=3 #Number of classes for segmentation
N_IMAGES = 200  #more than that available


images = "multiclass_saved_arrays/multiclass_datasets/_train_images.npy"
masks = "multiclass_saved_arrays/multiclass_datasets/_train_masks.npy"
#filename_images = np.load("multiclass_saved_arrays/_multiclass_filenames.npy") 

'''

print("Image data shape is: ", images.shape)
print("Mask data shape is: ", masks.shape)
print("Max pixel value in image is: ", images.max())
print("Labels in the mask are : ", np.unique(masks))

'''

COLORMAP = [
        [0, 0, 0], [37, 150, 190], [254,228,186]
    ]

CLASSES = [
    'background', 'mesh', 'wire'
    ]

for name, color in zip(CLASSES, COLORMAP):
    print(f"{name} - {color}")

    # background - [0, 0, 0]
    # mesh - [37, 150, 190]
    # wire - [254, 228, 186]


def process_mask(rgb_mask, colormap):
    output_mask = []

    for i, color in enumerate(colormap):
        cmap = np.all(np.equal(rgb_mask, color), axis=-1)
        output_mask.append(cmap)

    output_mask = np.stack(output_mask, axis=-1)
    return output_mask
    
#Processing the mask to one-hot mask
for x in masks:
    """ Reading the image and mask """
    mask = cv2.imread(x, cv2.IMREAD_COLOR)

    print(mask)

    """ Processing the mask to one-hot mask """
    processed_mask = process_mask(mask, COLORMAP)
    """ Converting one-hot mask to single channel mask """
    grayscale_mask = np.argmax(processed_mask, axis=-1)
    grayscale_mask = (grayscale_mask / len(CLASSES)) * 255
    grayscale_mask = np.expand_dims(grayscale_mask, axis=-1)

    for y in images : 

        image = cv2.imread(y, cv2.IMREAD_COLOR)

        """ Saving the image """
        line = np.ones((320, 5, 3)) * 255
        cat_images = np.concatenate([
        image, line, mask, line,
        np.concatenate([grayscale_mask, grayscale_mask, grayscale_mask], axis=-1)
        ], axis=1)

        cv2.imwrite(f"mask/{name}.png", cat_images)


