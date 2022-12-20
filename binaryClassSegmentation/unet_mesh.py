"""
Data augmentation
In the training set  images with rocks without mesh (clean walls)
Early stopping on mean iou
"""

EPOCHS = 250
PATIENCE = 50
BATCH_SIZE = 32
OPTIMIZER = 'adam'
DIM = 512
MODEL_NAME = "mesh_unet.h5"
CATEGORY = "mesh"

import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import cv2
import os
from evaluation import mean_iou_test, dice_coeff, pixel_accuracy, pixel_accuracy_class1

tf.random.set_seed(1)
src_arr = "/home/sakkaya/binary-segmentation/saved_arrays/" + CATEGORY

#Load train, validation and test data. The images are augmented in train and validation. For each image was created a random modified new image.
#Images are normalized in [0,1].
train_images = np.load(src_arr + "/" + CATEGORY + "_train_images.npy")
train_images = train_images/255.0

train_masks = np.load(src_arr + "/" + CATEGORY + "_train_masks.npy")
train_masks = np.expand_dims(train_masks, axis=-1)

val_images = np.load(src_arr + "/" + CATEGORY + "_val_images.npy")
val_images = val_images/255.0

val_masks = np.load(src_arr + "/" + CATEGORY + "_val_masks.npy")
val_masks = np.expand_dims(val_masks, axis=-1)

test_images = np.load(src_arr + "/" + CATEGORY + "_test_images.npy")
test_images = test_images/255.0

test_masks = np.load(src_arr + "/" + CATEGORY + "_test_masks.npy")
test_masks = np.expand_dims(test_masks, axis=-1)

print(train_images.shape)
print(train_masks.shape)

checkpoint_path = "/home/sakkaya/binary-segmentation/checkpoints" #where to save the model checkpoints
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

def unet(sz = (DIM, DIM, 3)):
  x = Input(sz)
  inputs = x
  
  #down sampling 
  f = 8
  layers = []
  
  for i in range(0, 6):
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    layers.append(x)
    x = MaxPooling2D() (x)
    f = f*2
  ff2 = 64 
  
  #bottleneck 
  j = len(layers) - 1
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
  x = Concatenate(axis=3)([x, layers[j]])
  j = j -1 
  
  #upsampling 
  for i in range(0, 5):
    ff2 = ff2//2
    f = f // 2 
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
    x = Concatenate(axis=3)([x, layers[j]])
    j = j -1 
    
  #classification 
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  outputs = Conv2D(1, 1, activation='sigmoid') (x)
  
  #model creation 
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
          ModelCheckpoint(checkpoint_path + '/'+ MODEL_NAME,  monitor="val_mean_iou", save_best_only=True, mode='max') #best on iou
    ]
    return checkpointer

model = unet(sz=(DIM,DIM,3))
model.summary()

"""## Training"""

model.fit(train_images, train_masks, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(val_images, val_masks), callbacks = build_callbacks())

"""## Results on test set

Qualitative measure: plot test image, prediction mask and target mask
"""
sample_dir = "/home/sakkaya/binary-segmentation/samples_" + CATEGORY #where to save the predictions
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

for i in range(test_images.shape[0]):
    test_img = test_images[i]
    test_mask = test_masks[i][:,:,0]
    combined = plot_img_and_masks(test_img, test_mask)
    cv2.imwrite(sample_dir + "/" + str(i) + '.jpg', combined * 255)

"""Quantitative measures"""
print("TEST RESULTS")
pred_masks = model.predict(test_images)

iou = mean_iou_test(test_masks, pred_masks)
print("mean iou", iou)
dice = dice_coeff(test_masks, pred_masks)
print("dice coeff.", dice)
acc = pixel_accuracy(test_masks, pred_masks)
print("pixel acc.", acc)
acc_1 = pixel_accuracy_class1(test_masks, pred_masks)
print("pixel acc. for " + CATEGORY + " only", acc_1)
