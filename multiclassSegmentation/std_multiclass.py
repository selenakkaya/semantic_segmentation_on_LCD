"""
target mask: (512,512,1)
output mask: one hot (512,512,3)
loss: SparseCategoricalCrossentropy
with dropout layers
"""

import numpy as np
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout
from keras.layers import Concatenate
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2



DIM = 512
EPOCHS = ...
PATIENCE = ...
BATCH_SIZE = ...
OPTIMIZER = tf.keras.optimizers....
N_CHANNELS = 3
N_CLASSES = 3
DIM = 512
MODEL_NAME = "unet_multiclass_std.h5"

#src_path = "saved_arrays_multiclass/"

src_path = "/.../"


train_images = np.load(src_path + "train_images.npy")
train_masks_one_hot = np.load(src_path + "train_masks.npy")
train_masks = np.argmax(train_masks_one_hot, axis=-1)
train_masks = np.expand_dims(train_masks, axis=-1)
print(train_masks.shape)


val_images = np.load(src_path + "val_images.npy")
val_masks_one_hot = np.load(src_path + "val_masks.npy")
val_masks = np.argmax(val_masks_one_hot, axis=-1)
val_masks = np.expand_dims(val_masks, axis=-1)
print(val_masks.shape)

test_images = np.load(src_path + "test_images.npy")
test_masks_one_hot = np.load(src_path + "test_masks.npy")
test_masks = np.argmax(test_masks_one_hot, axis=-1)
test_masks = np.expand_dims(test_masks, axis=-1)
print(test_masks.shape)

train_images = train_images/255.0
val_images = val_images/255.0
test_images = test_images/255.0 

print("train images", train_images.shape)
print("train masks", train_masks.shape)
print("val images", val_images.shape)
print("test images", test_images.shape)





checkpoint_path = "/.../"
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

    

def unet(sz, n_classes=3):
  x = Input(sz)
  inputs = x
  filter_size = ...

  
  #downsampling

  c1 = Conv2D(filter_size, 3, activation='...', padding = "...") (x)
  c1 = Dropout(0.1)(c1)
  c1 = Conv2D(filter_size, 3, activation='...', padding = "...") (c1)
  p1 = MaxPooling2D() (c1)


  c2 = Conv2D(filter_size*2, 3, activation='...', padding = "...") (p1)
  c2 = Dropout(0.1)(c2)
  c2 = Conv2D(filter_size*2, 3, activation='...', padding = "...") (c2)
  p2 = MaxPooling2D() (c2)


  c3 = Conv2D(filter_size*4, 3, activation='...', padding = "...") (p2)
  c3 = Dropout(0.2)(c3)
  c3 = Conv2D(filter_size*4, 3, activation='...', padding = "...") (c3)
  p3 = MaxPooling2D() (c3)


  c4 = Conv2D(filter_size*8, 3, activation='...', padding = "...") (p3)
  c4 = Dropout(0.2)(c4)
  c4 = Conv2D(filter_size*8, 3, activation='...', padding = "...") (c4)
  p4 = MaxPooling2D() (c4)


  #bottleneck 
  c5 = Conv2D(filter_size*16, 3, activation='...', padding = "...") (p4)
  c5 = Dropout(0.3)(c5)
  c5 = Conv2D(filter_size*16, 3, activation='...', padding = "...") (c5)
  

  #upsampling 


  u6 = Conv2DTranspose(filter_size*8, 2, strides=(2, 2))(c5)
  u6 = concatenate([u6,c4])
  c6 = Conv2D(filter_size*8, 3, activation='...', padding = "...") (u6)
  c6 = Dropout(0.2)(c6)
  c6 = Conv2D(filter_size*8, 3, activation='...', padding = "...") (c6)


  u7 = Conv2DTranspose(filter_size*4, 2, strides=(2, 2))(c6)
  u7 = concatenate([u7,c3])
  c7 = Conv2D(filter_size*4, 3, activation='...', padding = "...") (u7)
  c7 = Dropout(0.2)(c7)
  c7 = Conv2D(filter_size*4, 3, activation='...', padding = "...") (c7)


  u8 = Conv2DTranspose(filter_size*2, 2, strides=(2, 2))(c7)
  u8 = concatenate([u8,c2])
  c8 = Conv2D(filter_size*2, 3, activation='...', padding = "...") (u8)
  c8 = Dropout(0.1)(c8)
  c8 = Conv2D(filter_size*2, 3, activation='...', padding = "...") (c8)
  

  u9 = Conv2DTranspose(filter_size, 2, strides=(2, 2))(c8)
  u9 = concatenate([u9,c1], axis=3)
  c9 = Conv2D(filter_size, 3, activation='...', padding = "...") (u9)
  c9 = Dropout(0.1)(c9)
  c9 = Conv2D(filter_size, 3, activation='...', padding = "...") (c9)

  

    
  outputs = Conv2D(n_classes, 1,  activation='...') (c9)

  #model creation 
  model = Model(inputs=[inputs], outputs=[outputs])                

  return model



model = unet((DIM,DIM,3), n_classes=3)

model.compile(
    optimizer=OPTIMIZER,
    loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["..."],
)
model.summary()



def build_callbacks():
    checkpointer = [
          EarlyStopping(monitor="...", patience=PATIENCE, restore_best_weights=True),
          ModelCheckpoint(checkpoint_path + '/' + MODEL_NAME,  monitor="...", save_best_only=True)
    ]
    return checkpointer




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
plt.savefig('/home/sakkaya/best_multiclass/multiclass_std_unet_dropout.png')





def plot_img_and_masks(raw, mask):
    #mask is (512,512, 1)
   """...
   .
   ..."""

   

   
    return combined



sample_dir = "/..." #where to save the predictions

if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

for i in range(test_images.shape[0]):
    test_img = test_images[i]
    test_mask = test_masks[i]
    combined = plot_img_and_masks(test_img, test_mask)
    cv2.imwrite(sample_dir + "/" + str(i) + '.jpg', combined * 255)

"""Quantitative measures"""

from evaluation_multiclass import mean_iou_test, dice_coeff, pixel_accuracy

print("TEST RESULTS for multiclass segmentation std unet dropot multiclass")
pred_masks_one_hot = model.predict(test_images) #(n examples,h,w,3)

#test_masks1 = np.argmax(test_masks, axis=-1) #(n examples,h,w)
pred_masks = np.argmax(pred_masks_one_hot, axis=-1)

iou, iou_classes = mean_iou_test(test_masks, pred_masks,num_classes=3)
print("mean iou", iou)
print(iou_classes)
dice, dice_classes = dice_coeff(test_masks, pred_masks,num_classes=3)
print("dice coeff.", dice)
print(dice_classes)
acc, acc_classes = pixel_accuracy(test_masks, pred_masks,num_classes=3)
print("pixel acc.", acc)
print(acc_classes)