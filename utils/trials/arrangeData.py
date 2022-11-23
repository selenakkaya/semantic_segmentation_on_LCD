import tensorflow as tf
import numpy as np


seed = 42
np.random.seed = seed


CATEGORY = "mesh" #mesh or wire
SIZE = 512
TRAIN_SIZE = 0.85

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

MESH_IMG_PATH = 'saved_arrays/datasets/mesh_images.npy'
MESH_MASK_PATH = 'saved_arrays/datasets/mesh_masks.npy'


image_dataset = np.load(MESH_IMG_PATH)
mask_dataset = np.load(MESH_MASK_PATH)


#Normalize images
from keras.utils import normalize

image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1),3)

#Do not normalize masks, just rescale to 0 to 1.
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 0)



from simple_unet_model import simple_unet_model   #Use normal unet model

def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()