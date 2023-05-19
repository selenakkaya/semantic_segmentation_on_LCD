EPOCHS = 250
PATIENCE = 50
BATCH_SIZE = 10
DIM = 512
MODEL_NAME = "standard_unet_aug_mesh_64.h5"
CATEGORY = "mesh" #wire or mesh

import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, concatenate, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import cv2
import os
from evaluation import mean_iou_test, dice_coeff, pixel_accuracy, pixel_accuracy_one_class

tf.random.set_seed(1)
src_arr = "/home/sakkaya/binary-segmentation/saved_arrays/" + CATEGORY # HPC
#src_arr = "../saved_arrays/" + CATEGORY # local pc


#Load train, validation and test data. The images are augmented in train and validation. For each image was created a random modified new image.
#Images are normalized in [0,1].
train_images = np.load(src_arr + "/" + CATEGORY + "_aug_train_images.npy")
train_images = train_images/255.0

train_masks = np.load(src_arr + "/" + CATEGORY + "_aug_train_masks.npy")
train_masks = np.expand_dims(train_masks, axis=-1)

val_images = np.load(src_arr + "/" + CATEGORY + "_aug_val_images.npy")
val_images = val_images/255.0

val_masks = np.load(src_arr + "/" + CATEGORY + "_aug_val_masks.npy")
val_masks = np.expand_dims(val_masks, axis=-1)

test_images = np.load(src_arr + "/" + CATEGORY + "_test_images.npy")
test_images = test_images/255.0

test_masks = np.load(src_arr + "/" + CATEGORY + "_test_masks.npy")
test_masks = np.expand_dims(test_masks, axis=-1)

print(train_images.shape)
print(train_masks.shape)


#checkpoint_path = "../checkpoints/standard_softmax" #where to save the model checkpoints

checkpoint_path = "/home/sakkaya/std_unet/mesh/checkpoints_std_aug_"+CATEGORY #where to save the model checkpoints

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

  
  #downsampling

  c1 = Conv2D(64, 3, activation='relu', padding = "same") (x)
  c1 = Conv2D(64, 3, activation='relu', padding = "same") (c1)
  p1 = MaxPooling2D() (c1)


  c2 = Conv2D(128, 3, activation='relu', padding = "same") (p1)
  c2 = Conv2D(128, 3, activation='relu', padding = "same") (c2)
  p2 = MaxPooling2D() (c2)


  c3 = Conv2D(256, 3, activation='relu', padding = "same") (p2)
  c3 = Conv2D(256, 3, activation='relu', padding = "same") (c3)
  p3 = MaxPooling2D() (c3)


  c4 = Conv2D(512, 3, activation='relu', padding = "same") (p3)
  c4 = Conv2D(512, 3, activation='relu', padding = "same") (c4)
  p4 = MaxPooling2D() (c4)


  #bottleneck 
  c5 = Conv2D(1024, 3, activation='relu', padding = "same") (p4)
  c5 = Conv2D(1024, 3, activation='relu', padding = "same") (c5)
  

  #upsampling 


  u6 = Conv2DTranspose(512, 2, strides=(2, 2))(c5)
  u6 = concatenate([u6,c4])
  c6 = Conv2D(512, 3, activation='relu', padding = "same") (u6)
  c6 = Dropout(0.2)(c6)
  c6 = Conv2D(512, 3, activation='relu', padding = "same") (c6)


  u7 = Conv2DTranspose(256, 2, strides=(2, 2))(c6)
  u7 = concatenate([u7,c3])
  c7 = Conv2D(256, 3, activation='relu', padding = "same") (u7)
  c7 = Conv2D(256, 3, activation='relu', padding = "same") (c7)


  u8 = Conv2DTranspose(128, 2, strides=(2, 2))(c7)
  u8 = concatenate([u8,c2])
  c8 = Conv2D(128, 3, activation='relu', padding = "same") (u8)
  c8 = Conv2D(128, 3, activation='relu', padding = "same") (c8)
  

  u9 = Conv2DTranspose(64, 2, strides=(2, 2))(c8)
  u9 = concatenate([u9,c1], axis=3)
  c9 = Conv2D(64, 3, activation='relu', padding = "same") (u9)
  c9 = Conv2D(64, 3, activation='relu', padding = "same") (c9)

  

    
  outputs = Conv2D(1, (1, 1),  activation='sigmoid') (c9)

  #model creation 
  model = Model(inputs=[inputs], outputs=[outputs])
  model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = [mean_iou])
                

  """## Training"""

  history = model.fit(train_images, train_masks, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(val_images, val_masks), callbacks = build_callbacks())
 
  #training and validation loss
  import matplotlib.pyplot as plt

  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(1, len(loss) + 1)

  plt.plot(epochs, loss, 'bo', label='Training Loss')
  plt.plot(epochs, val_loss, 'r', label='Validation Loss')
  plt.title('Training and Validation Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()
  plt.savefig('/home/sakkaya/std_unet/mesh/loss_std_unet_aug_mesh.png')

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




       
"""## Results on test set

Qualitative measure: plot test image, prediction mask and target mask
"""
sample_dir = "/home/sakkaya/std_unet/mesh/samples_std_unet_aug_" + CATEGORY #where to save the predictions
#sample_dir = "samples_unet_mesh"
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

for i in range(test_images.shape[0]):
    test_img = test_images[i]
    test_mask = test_masks[i][:,:,0]
    combined = plot_img_and_masks(test_img, test_mask)
    cv2.imwrite(sample_dir + "/" + str(i) + '.jpg', combined * 255)



"""Quantitative measures"""
print("TEST RESULTS for mesh std aug unet ")
pred_masks = model.predict(test_images)

iou = mean_iou_test(test_masks, pred_masks)
print("mean iou", iou)
dice = dice_coeff(test_masks, pred_masks)
print("dice coeff.", dice)
acc = pixel_accuracy(test_masks, pred_masks)
print("pixel acc.", acc)
acc_1 = pixel_accuracy_one_class(test_masks, pred_masks)
print("pixel acc. for " + CATEGORY + " only", acc_1)
