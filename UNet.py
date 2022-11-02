import tensorflow as tf

IMG_WIDTH = 128
IMG_HEIGHT = 128
CHANNELS = 3

##################################################################################################
###############################_________Build the model_________##################################
##################################################################################################


inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, CHANNELS)) 

#convert input pixel values into floating point
s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)

#downsample path

C1 = tf.keras.layers.Conv2D( 16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same" )(s)
C1 = tf.keras.layers.Dropout(0.1)(C1)
C1 = tf.keras.layers.Conv2D( 16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same" )(C1)
P1 = tf.keras.layers.MaxPool2D((2, 2))(C1)

C2 = tf.keras.layers.Conv2D( 32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same" )(P1)
C2 = tf.keras.layers.Dropout(0.1)(C2)
C2 = tf.keras.layers.Conv2D( 32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same" )(C2)
P2 = tf.keras.layers.MaxPool2D((2, 2))(C2)

C3 = tf.keras.layers.Conv2D( 64 , (3, 3), activation="relu", kernel_initializer="he_normal", padding="same" )(P2)
C3 = tf.keras.layers.Dropout(0.2)(C3)
C3 = tf.keras.layers.Conv2D( 64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same" )(C3)
P3 = tf.keras.layers.MaxPool2D((2, 2))(C3)

C4 = tf.keras.layers.Conv2D( 128 , (3, 3), activation="relu", kernel_initializer="he_normal", padding="same" )(P3)
C4 = tf.keras.layers.Dropout(0.2)(C4)
C4 = tf.keras.layers.Conv2D( 128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same" )(C4)
P4 = tf.keras.layers.MaxPool2D(pool_size = (2, 2))(C4)

#bottleneck
C5 = tf.keras.layers.Conv2D( 256 , (3, 3), activation="relu", kernel_initializer="he_normal", padding="same" )(P4)
C5 = tf.keras.layers.Dropout(0.3)(C5)
C5 = tf.keras.layers.Conv2D( 256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same" )(C5)

#upsample path

"""U6=U6+C4"""
U6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(C5)
U6 = tf.keras.layers.concatenate([U6, C4])
C6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(U6)
C6 = tf.keras.layers.Dropout(0.2)(C6)
C6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(C6)
 
"""U7=U7+C3"""
U7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(C6)
U7 = tf.keras.layers.concatenate([U7, C3])
C7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(U7)
C7 = tf.keras.layers.Dropout(0.2)(C7)
C7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(C7)
 
"""U8=U8+C2"""
U8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(C7)
U8 = tf.keras.layers.concatenate([U8, C2])
C8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(U8)
C8 = tf.keras.layers.Dropout(0.1)(C8)
C8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(C8)
 
"""U9=U9+C1"""
U9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(C8)
U9 = tf.keras.layers.concatenate([U9, C1], axis=3)
C9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(U9)
C9 = tf.keras.layers.Dropout(0.1)(C9)
C9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(C9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(C9)

model = tf.keras.Model( inputs = [inputs], outputs = [outputs])
model.compile(optimizer = "adam", loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
