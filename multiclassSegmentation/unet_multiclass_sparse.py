"""
target mask: (512,512,1)
output mask: one hot (512,512,3)
loss: SparseCategoricalCrossentropy
"""

import numpy as np
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers import Concatenate
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2



DIM = 512
EPOCHS = 500
PATIENCE = 50
BATCH_SIZE = 1
OPTIMIZER = "adam"
N_CHANNELS = 3
N_CLASSES = 3
DIM = 512
MODEL_NAME = "unet_multiclass.h5"


train_images = np.load("val_images.npy")[0:1]
train_masks_one_hot = np.load("val_masks.npy")[0:1]

train_masks = np.argmax(train_masks_one_hot, axis=-1)
print(train_masks)

train_masks = np.expand_dims(train_masks, axis=-1)
print(train_masks.shape)


val_images = train_images
val_masks = train_masks
test_images = train_images
test_masks = train_masks
print("train images", train_images.shape)
print("train masks", train_masks.shape)
print("val images", val_images.shape)
print("test images", test_images.shape)


train_images = train_images/255.0
val_images = val_images/255.0
test_images = test_images/255.0

checkpoint_path = "checkpoints"

if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)

def unet(sz, n_classes=3):
    x = Input(sz)
    inputs = x
    # down sampling
    f = 8
    layers = []

    for i in range(0, 6):
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        layers.append(x)
        x = MaxPooling2D()(x)
        f = f * 2
    ff2 = 64

    # bottleneck
    j = len(layers) - 1
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same')(x)
    x = Concatenate(axis=3)([x, layers[j]])
    j = j - 1

    # upsampling
    for i in range(0, 5):
        ff2 = ff2 // 2
        f = f // 2
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same')(x)
        x = Concatenate(axis=3)([x, layers[j]])
        j = j - 1

    # classification
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    outputs = Conv2D(n_classes, 1, activation='softmax')(x)
    #outputs = Conv2D(n_classes, 1)(x)

    # model creation
    model = Model(inputs=[inputs], outputs=[outputs])

    return model


model = unet((DIM,DIM,3), n_classes=3)
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.compile(
    optimizer=OPTIMIZER,
    loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)
model.summary()


def build_callbacks():
    checkpointer = [
          EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
          ModelCheckpoint(checkpoint_path + '/' + MODEL_NAME,  monitor="val_loss", save_best_only=True)
    ]
    return checkpointer

model.fit(train_images, train_masks,  batch_size=BATCH_SIZE, verbose=1, epochs=EPOCHS, validation_data=(val_images, val_masks), callbacks = build_callbacks())


def plot_img_and_masks(raw, mask):
    #mask is (512,512, 1)
    pred_one_hot = model.predict(np.epand_dims(raw, 0))
    pred = np.argmax(pred_one_hot, axis=-1)[0,:,:]
    pred_msk = np.stack((pred,) * 3, axis=-1)
    pred_msk[(pred_msk[:, :, 0] == 1) & (pred_msk[:, :, 1] == 1) & (pred_msk[:, :, 2] == 1)] = (255,0,0)
    pred_msk[(pred_msk[:, :, 0] == 2) & (pred_msk[:, :, 1] == 2) & (pred_msk[:, :, 2] == 2)] = (0,255,0)

    #mask1 = np.argmax(mask, axis=-1)
    target_msk = np.stack((mask[:,:,0],) * 3, axis=-1)
    print("target_msk", target_msk.shape)
    target_msk[(target_msk[:, :, 0] == 1) & (target_msk[:, :, 1] == 1) & (target_msk[:, :, 2] == 1)] = (255,0,0)
    target_msk[(target_msk[:, :, 0] == 2) & (target_msk[:, :, 1] == 2) & (target_msk[:, :, 2] == 2)] = (0,255,0)

    raw = np.float32(raw)
    pred_msk = np.float32(pred_msk)
    target_msk = np.float32(target_msk)
    raw = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    pred_msk = cv2.cvtColor(pred_msk, cv2.COLOR_RGB2BGR)
    target_msk = cv2.cvtColor(target_msk, cv2.COLOR_RGB2BGR)
    combined = np.concatenate([raw, pred_msk, target_msk], axis=1)
    return combined



sample_dir = "samples_multiclass" #where to save the predictions

if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

for i in range(test_images.shape[0]):
    test_img = test_images[i]
    test_mask = test_masks[i]
    combined = plot_img_and_masks(test_img, test_mask)
    cv2.imwrite(sample_dir + "/" + str(i) + '.jpg', combined * 255)

"""Quantitative measures"""

from evaluation_multiclass import mean_iou_test, dice_coeff, pixel_accuracy

print("TEST RESULTS for multiclass segmentation")
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



