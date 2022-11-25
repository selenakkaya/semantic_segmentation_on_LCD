"""
Data augmentation: 1 image changing in illumination and (50%) flipping + 1 image randomly changing order of channels and (50%) flipping
--> training set is tripled
In the training set images with rocks without mesh (clean walls)
Early stopping on mean iou
"""

EPOCHS = 250
PATIENCE = 50
BATCH_SIZE = 32
OPTIMIZER = 'rmsprop'
DIM = 512
MODEL_NAME = "mesh_unet_cw_plus.h5"



import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import cv2
import os
from evaluation import mean_iou_test, dice_coeff, pixel_accuracy

tf.random.set_seed(1)
src_arr = "../processing/saved_arrays_all"

#Load train, validation and test data. The images are augmented in train and validation.
#Images are normalized in [0,1].
train_images = np.load(src_arr + "/aug_train_images.npy")
cw_images = np.load(src_arr + "/cw_images.npy") #images without mesh
train_images = np.concatenate((train_images, cw_images), axis=0)[:5]
train_images = train_images/255.0
train_masks = np.load(src_arr + "/aug_train_masks_mesh.npy")
cw_masks = np.load(src_arr + "/cw_masks.npy")
train_masks = np.concatenate((train_masks, cw_masks), axis=0)[:5]
train_masks = np.expand_dims(train_masks, axis=-1)

val_images = np.load(src_arr + "/aug_val_images.npy")[:5]
val_images = val_images/255.0
val_masks = np.load(src_arr + "/aug_val_masks_mesh.npy")[:5]
val_masks = np.expand_dims(val_masks, axis=-1)

test_images = np.load(src_arr + "/test_images.npy")[:5]
test_images = test_images/255.0
test_masks = np.load(src_arr + "/test_masks_mesh.npy")[:5]
test_masks = np.expand_dims(test_masks, axis=-1)

print(train_images.shape)
print(train_masks.shape)

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

def unet(sz):
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
sample_dir = "samples_mesh_cw_plus" #where to save the predictions
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
