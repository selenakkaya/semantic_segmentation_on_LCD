import numpy as np
import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate
import random
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import cv2
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping



DIM = 512
EPOCHS = 250
PATIENCE = 50
BATCH_SIZE = 32
OPTIMIZER = 'adam'
N_CLASSES = 3
DIM = 512
MODEL_NAME = "unet_multiclass.h5"

src_path = "/home/sakkaya/multiclass_segmentation/saved_arrays_multiclass"
train_images = np.load(src_path + "/train_images.npy")[:10] #FIXME aggiungere tutte le immagini
train_masks = np.load(src_path + "/train_masks.npy")[:10]
val_images = np.load(src_path + "/val_images.npy")[:10]
val_masks = np.load(src_path + "/val_masks.npy")[:10]
test_images = np.load(src_path + "/test_images.npy")[:10]
test_masks = np.load(src_path + "/test_masks.npy")[:10]
print("train images", train_images.shape)
print("train masks", train_masks.shape)
print("val images", val_images.shape)
print("test images", test_images.shape)

dst_dir = "/home/sakkaya/multiclass_segmentation/samples_multiclass/during_training"
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

train_images = train_images/255.0
val_images = val_images/255.0
test_images = test_images/255.0

checkpoint_path = "/home/sakkaya/multiclass_segmentation/checkpoints" #where to save the model checkpoints
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)





from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate



def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)  #Not in the original network
    x = Activation("relu")(x)

    return x

    
#Encoder block: Conv block followed by maxpooling


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p   
    
#Decoder block
#skip features gets input from encoder for concatenation

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

#Build Unet using the blocks
def build_unet(input_shape, n_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024) #Bridge

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    if n_classes == 1:  #Binary
      activation = 'sigmoid'
    else:
      activation = 'softmax'

    outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d4)  #Change the activation based on n_classes
    print(activation)

    model = Model(inputs, outputs, name="U-Net")
    return model






model = build_unet((DIM,DIM,3), n_classes=3)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(train_images, train_masks, 
                    batch_size = BATCH_SIZE , 
                    verbose=1, 
                    epochs= EPOCHS, 
                    validation_data=(val_images, val_masks), 
                    shuffle=False)


#Save the model for future use
model.save('/home/sakkaya/multiclass_segmentation/multiclass_unet.hdf5')

#Load previously saved model
from keras.models import load_model
model = load_model("/home/sakkaya/multiclass_segmentation/multiclass_unet.hdf5", compile=False)

y_pred=model.predict(test_images)
y_pred.shape

y_pred_argmax=np.argmax(y_pred, axis=3)
y_pred_argmax.shape

#Using built in keras function
from keras.metrics import MeanIoU
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(test_masks[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

