import numpy as np

import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import cv2
import os

EPOCHS = 200
PATIENCE = 50
BATCH_SIZE = 12
OPTIMIZER = 'adam'
DIM = 512
MODEL_NAME = "best_gridsearch_unet_mesh.h5"
CATEGORY = "mesh"

#tf.random.set_seed(1)

#src_arr = "saved_arrays/" + CATEGORY # local pc
src_arr = "/home/sakkaya/binary-segmentation/saved_arrays/" + CATEGORY #hpc


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
print(val_masks.shape)
print(test_images.shape)



checkpoint_path = "/home/sakkaya/grid_search/checkpoints_best" + CATEGORY  #where to save the model checkpoints
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

def binary_focal_loss(gamma=2., alpha=.25):
   
    def binary_focal_loss_fixed(y_true, y_pred):
       
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed

def unet(optimizer, loss, sz = (DIM, DIM, 3)):
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
    model.compile(optimizer = optimizer, loss = loss, metrics = [mean_iou])
    model.summary()

    return model


def build_callbacks():
    checkpointer = [
          EarlyStopping(monitor="val_mean_iou", patience=PATIENCE, restore_best_weights=True, mode="max"),
          ModelCheckpoint(checkpoint_path + '/'+ MODEL_NAME,  monitor="val_mean_iou", save_best_only=True, mode='max') #best on iou
    ]
    return checkpointer


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


from evaluation_updated import mean_iou_test, dice_coeff, pixel_accuracy, pixel_accuracy_class1

def quantitative_measures(pred_masks):
    test_masks = np.load(src_arr + "/" + CATEGORY + "_test_masks.npy")
    test_masks = np.expand_dims(test_masks, axis=-1)
    print("TEST RESULTS")

    iou = mean_iou_test(test_masks, pred_masks)
    print("mean iou", iou)
    dice = dice_coeff(test_masks, pred_masks)
    print("dice coeff.", dice)
    acc = pixel_accuracy(test_masks, pred_masks)
    print("pixel acc.", acc)
    acc_1 = pixel_accuracy_class1(test_masks, pred_masks)
    print("pixel acc. for " + CATEGORY + " only", acc_1)
    
    return iou, dice, acc, acc_1


#-------------------GRID SEARCH----------------
import itertools


optimizers = ["adam", "adadelta", "sgd", "adagrad", "rmsprop"]
losses = [binary_focal_loss(gamma=2.0, alpha=0.25), "binary_crossentropy"]
#strides = [(2,2), (3,3)]
#paddings = ["same", "valid" ]
combinations = list(itertools.product(optimizers, losses))


result_list = []
for optimizer, loss in combinations:
    print("Optimizer:", optimizer, "Loss function:", loss)
    model = unet(optimizer, loss, sz=(DIM,DIM,3))
    history  = model.fit(train_images, train_masks, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(val_images, val_masks), callbacks = build_callbacks())
    pred_masks = model.predict(test_images)
    iou, dice, acc, acc_1 = quantitative_measures(pred_masks)
    result = [optimizer, loss, history.history, iou, dice, acc, acc_1]
    result_list.append(result)
    
    # Write the results to a .txt file
    with open('combination_results.txt', 'w') as f:
        for result in result_list:
            f.write("Optimizer: " + result[0] + ", Loss function: " + str(result[1]) + ", quantitative results :  iou, dice, acc, acc_1 " + "\n")
            for i in range(EPOCHS):
                f.write("Epoch " + str(i+1) + ": " + "loss: " + str(result[2]['loss'][i]) + ", val_loss: " + str(result[2]['val_loss'][i])  + ", iou " +  str(result[3]) + ",dice "  +  str(result[4])  + ",acc " +   str(result[5]) + ",acc_1 " +   str(result[6]) + "\n")


   