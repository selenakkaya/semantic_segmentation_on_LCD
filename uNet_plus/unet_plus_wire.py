#set parametrs
EPOCHS =
PATIENCE = 50
BATCH_SIZE =
OPTIMIZER = 'rmsprop' #or adam
DIM = 512
MODEL_NAME =

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


#TODO load images and masks for train val and test. Normalize images in [0,1]. Expand dims of masks in the last channel
#choose if adding augmented images or not

sample_dir = "samples_wire" #where to save the predictions
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

checkpoint_path = "checkpoints" #where to save the model checkpoints
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

#during training
def mean_iou(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
    return iou

def unet_plus():
  #TODO write the unet++

  model = Model(inputs=[inputs], outputs=[outputs])
  model.compile(optimizer = OPTIMIZER, loss = 'binary_crossentropy', metrics = [mean_iou])
  
  return model

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


model = unet_plus(sz=(DIM,DIM,3))
model.summary()

"""## Training"""

model.fit(train_images, train_masks, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(val_images, val_masks), callbacks = build_callbacks())

"""## Results on test set

Qualitative measure: plot test image, prediction mask and target mask
"""

for i in range(test_images.shape[0]):
    test_img = test_images[i]
    test_mask = test_masks[i][:,:,0]
    combined = plot_img_and_masks(test_img, test_mask)
    cv2.imwrite(sample_dir + "/" + str(i) + '.jpg', combined * 255)


"""Quantitative measures"""
pred_masks = model.predict(test_images)

#TODO print the four measures on test set