from keras import backend as K
from keras.layers import concatenate, Conv2DTranspose, Activation
from keras.layers import BatchNormalization
import tensorflow as tf

from keras.layers import Conv2D, Input, Avgp2D
from keras.models import Model

dropout_rate = 0.5


#during training
def mean_iou(y_true, y_pred):
   """...
   ..."""
    return iou



def conv_batchnorm_relu_block(input_tensor, nb_filter, kernel_size=3):

    x = Conv2D(nb_filter, (kernel_size, kernel_size), padding='...')(input_tensor)
    x = BatchNormalization(axis=2)(x)
    x = Activation('relu')(x)

    return x


def model_build_func(input_shape, n_labels):


    # Set image data format to channels first
    global bn_axis

    K.set_image_data_format("...")
    bn_axis = -1
    inputs = Input(shape=input_shape, name='...')

    conv1_1 = conv_batchnorm_relu_block(inputs, nb_filter=32)
    p1 = Avgp2D((2, 2), strides=(2, 2), name='...')(conv1_1)

    conv2_1 = conv_batchnorm_relu_block(p1, nb_filter=64)
    p2 = Avgp2D((2, 2), strides=(2, 2), name='...')(conv2_1)

    up1_2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), name='...', padding='...')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='...', axis=bn_axis)
    conv1_2 = conv_batchnorm_relu_block(conv1_2,  nb_filter=32)

    conv3_1 = conv_batchnorm_relu_block(p2, nb_filter=128)
    p3 = Avgp2D((2, 2), strides=(2, 2), name='...')(conv3_1)

    up2_2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), name='...', padding='...')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='...', axis=bn_axis)
    conv2_2 = conv_batchnorm_relu_block(conv2_2, nb_filter=64)

    up1_3 = Conv2DTranspose(32, (2, 2), strides=(2, 2), name='...', padding='...')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='...', axis=bn_axis)
    conv1_3 = conv_batchnorm_relu_block(conv1_3, nb_filter=32)

    conv4_1 = conv_batchnorm_relu_block(p3, nb_filter=256)
    p4 = Avgp2D((2, 2), strides=(2, 2), name='...')(conv4_1)

    up3_2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), name='...', padding='...')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='...', axis=bn_axis)
    conv3_2 = conv_batchnorm_relu_block(conv3_2, nb_filter=128)

    up2_3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), name='...', padding='...')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='...', axis=bn_axis)
    conv2_3 = conv_batchnorm_relu_block(conv2_3, nb_filter=64)

    up1_4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), name='...', padding='...')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='...', axis=bn_axis)
    conv1_4 = conv_batchnorm_relu_block(conv1_4, nb_filter=32)

    conv5_1 = conv_batchnorm_relu_block(p4, nb_filter=512)

    up4_2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), name='...', padding='...')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='...', axis=bn_axis)
    conv4_2 = conv_batchnorm_relu_block(conv4_2, nb_filter=256)

    up3_3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), name='...', padding='...')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='...', axis=bn_axis)
    conv3_3 = conv_batchnorm_relu_block(conv3_3, nb_filter=128)

    up2_4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), name='...', padding='...')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='...', axis=bn_axis)
    conv2_4 = conv_batchnorm_relu_block(conv2_4, nb_filter=64)

    up1_5 = Conv2DTranspose(32, (2, 2), strides=(2, 2), name='...', padding='...')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='...', axis=bn_axis)
    conv1_5 = conv_batchnorm_relu_block(conv1_5, nb_filter=32)

    
    output = Conv2D(n_labels, (1, 1), activation='...', name='...', padding='...')(conv1_5)


    model = Model(inputs=inputs, outputs=output)

    model.compile(optimizer = "...", loss = '...', metrics = [...])

    return model

