"""
Data augmentation for mesh or wire.
From an image create a random modified image.
Change illumination, sometimes horizontal flipping and gaussian noise.
For flipped images also the mask is flipped
"""

import numpy as np
import cv2
import random
import os

random.seed(1)


def horizontal_flip(img):
    horizontal_img = cv2.flip(img.copy(), 1)
    return horizontal_img

def illumination_changing(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image.copy(), table)

def gaussian_noise(img, k=1):
    gauss = np.random.normal(0, k, img.size)
    gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype('uint8')
    img_gauss = cv2.add(img, gauss)
    return img_gauss


dst_dir_arr = "saved_arrays"
if not os.path.exists(dst_dir_arr):
    os.mkdir(dst_dir_arr)

train_images = np.load("train_images.npy")
train_masks = np.load("train_masks.npy")
val_images = np.load("val_images.npy")
val_masks = np.load("val_masks.npy")

print("train_images", train_images.shape)
print("val_images", val_images.shape)


def get_augmented_image(img, mask):
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    aug_mask = mask.copy()

    rand_ill = random.uniform(0.4, 1.5)
    aug_img = illumination_changing(img, rand_ill)

    # if adding gaussian noise (50% of probability)
    rand_noise = random.uniform(0.0, 1.0)
    if rand_noise > 0.5:
        aug_img = gaussian_noise(img, 0.4)

    # if flipping the image
    if rand_ill > 0.7 and rand_ill < 1.3:  # if the illumination change is irrelevant at least flip the image
        rand_flip = 1
    else:
        rand_flip = random.uniform(0.0, 1.0)
    if rand_flip > 0.5:
        aug_img = horizontal_flip(aug_img)
        aug_mask = cv2.flip(aug_mask, 1) #flip the mask

    return aug_img, aug_mask


aug_train_images = []
aug_train_masks = []
for i in range(train_images.shape[0]):
    img = train_images[i]
    mask = train_masks[i]
    aug_train_images.append(img)
    aug_train_masks.append(mask)
    aug_img, aug_mask = get_augmented_image(img, mask)
    aug_train_images.append(aug_img)
    aug_train_masks.append(aug_mask)


aug_val_images = []
aug_val_masks = []
for i in range(val_images.shape[0]):
    img = val_images[i]
    mask = val_masks[i]
    aug_val_images.append(img)
    aug_val_masks.append(mask)
    aug_img, aug_mask = get_augmented_image(img, mask)
    aug_val_images.append(aug_img)
    aug_val_masks.append(aug_mask)

aug_train_images = np.array(aug_train_images)
aug_train_masks = np.array(aug_train_masks)
aug_val_images = np.array(aug_val_images)
aug_val_masks = np.array(aug_val_masks)

print("aug_train_images", aug_train_images.shape)
print("aug_train_masks", aug_train_masks.shape)
print("aug_val_images", aug_val_images.shape)
print("aug_val_masks", aug_val_masks.shape)

np.save("saved_arrays/aug_train_images.npy", aug_train_images)
np.save("saved_arrays/aug_train_masks.npy", aug_train_masks)
np.save("saved_arrays/aug_val_images.npy", aug_val_images)
np.save("saved_arrays/aug_val_masks.npy", aug_val_masks)

"""
for i in range(aug_train_images.shape[0]):
    img = aug_train_images[i]
    msk = aug_train_masks[i]
    cv2.imwrite(dst_dir_arr + "/" + str(i) + '.jpg', img * 255)
"""