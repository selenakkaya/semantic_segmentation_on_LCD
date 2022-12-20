#set parametrs
EPOCHS = 250
PATIENCE = 50
BATCH_SIZE = 8
OPTIMIZER = 'adam' #or rmsprop
DIM = 512
MODEL_NAME = 'mesh_unet_plus_wire.h5'

CATEGORY = 'wire'


import numpy as np
import random
import cv2
import os
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from evaluation import mean_iou_test, dice_coeff, pixel_accuracy, pixel_accuracy_class1


tf.random.set_seed(1)
src_arr = '/home/sakkaya/UNetPlus/saved_arrays' 


#load images and masks for train val and test. Normalize images in [0,1]. Expand dims of masks in the last channel
#choose if adding augmented images or not


train_images = np.load(src_arr + "/" + CATEGORY + "/wire_train_images.npy")
train_images = train_images/255.0

train_masks = np.load(src_arr + "/" + CATEGORY + "/wire_train_masks.npy")
train_masks = np.expand_dims(train_masks, axis=-1)

val_images = np.load(src_arr + "/" + CATEGORY + "/wire_val_images.npy")
val_images = val_images/255.0
val_masks = np.load(src_arr + "/" + CATEGORY + "/wire_val_masks.npy")
val_masks = np.expand_dims(val_masks, axis=-1)

test_images = np.load(src_arr + "/" + CATEGORY + "/wire_test_images.npy")
test_images = test_images/255.0
test_masks = np.load(src_arr + "/" + CATEGORY + "/wire_test_masks.npy")
test_masks = np.expand_dims(test_masks, axis=-1)

print(train_images.shape)
print(train_masks.shape)



sample_dir = "/home/sakkaya/UNetPlus/samples_" + CATEGORY #where to save the predictions
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

checkpoint_path = "/home/sakkaya/UNetPlus/checkpoints" #where to save the model checkpoints
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)





def plot_img_and_masks(raw, mask):
    # image + prediction mask + target mask
    pred = model.predict(np.expand_dims(raw, 0))
    pred_msk = pred.squeeze()
    pred_msk = np.stack((pred_msk,) * 3, axis=-1)
    pred_msk[pred_msk >= 0.5] = 1
    pred_msk[pred_msk < 0.5] = 0
    target_msk = np.stack((mask,) * 3, axis=-1)
    raw = np.float32(raw)
    raw = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    combined = np.concatenate([raw, pred_msk, target_msk], axis=1)
    return combined



def build_callbacks():
    checkpointer = [
          EarlyStopping(monitor="val_mean_iou", patience=PATIENCE, restore_best_weights=True, mode="max"),
          ModelCheckpoint(checkpoint_path + '/' + MODEL_NAME, monitor="val_mean_iou", save_best_only=True, mode='max')
    ]
    return checkpointer

from unet_plus_model import model_build_func
model = model_build_func((DIM,DIM,3), 1, False)
model.summary()

"""## Training"""

model.fit(train_images, train_masks, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(val_images, val_masks), callbacks = build_callbacks())

"""## Results on test set

Qualitative measure: plot test image, prediction mask and target mask
"""

""
sample_dir = "UNetP_samples_" + CATEGORY #where to save the predictions
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

for i in range(test_images.shape[0]):
    test_img = test_images[i]
    test_mask = test_masks[i][:,:,0]
    combined = plot_img_and_masks(test_img, test_mask)
    cv2.imwrite(sample_dir + "/" + str(i) + '.jpg', combined * 255)


"""Quantitative measures"""
print("TEST RESULTS: " + CATEGORY +" for unet++")

pred_masks = model.predict(test_images)

#TODO print the four measures on test set

iou = mean_iou_test(test_masks, pred_masks)
print("mean iou", iou)
dice = dice_coeff(test_masks, pred_masks)
print("dice coeff.", dice)
acc = pixel_accuracy(test_masks, pred_masks)
print("pixel acc.", acc)
acc_for_wire = pixel_accuracy_class1(test_masks, pred_masks)
print("pixel acc. for wire only", acc_for_wire)
