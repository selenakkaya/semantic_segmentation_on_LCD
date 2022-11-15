import random
import numpy as np

random.seed(1)


CATEGORY = "multiclass" 
SIZE = 512
TRAIN_SIZE = 0.75
VAL_SIZE = 0.15

images = np.load("multiclass_saved_arrays/_" + CATEGORY + "_images.npy")
masks = np.load("multiclass_saved_arrays/_" + CATEGORY + "_masks.npy")
filenames = np.load("multiclass_saved_arrays/_" + CATEGORY + "_filenames.npy")

print(filenames)

site1 = []
site2 = []
site3 = []


for filename in filenames:
    fileN= int(filename.split('_')[1][3:])

    if(fileN < 173 and fileN not in site1):
        site1.append(fileN)
    elif(fileN >= 173 and fileN < 286 and fileN not in site2 ):
        site2.append(fileN)
    elif (fileN >= 286 and fileN not in site3):
        site3.append(fileN)



n_train_1 = int(len(site1)*TRAIN_SIZE)
n_train_2 = int(len(site2)*TRAIN_SIZE)
n_train_3 = int(len(site3)*TRAIN_SIZE)

n_val_1 = int(len(site1)*VAL_SIZE)
n_val_2 = int(len(site2)*VAL_SIZE)
n_val_3 = int(len(site3)*VAL_SIZE)

random.shuffle(site1)
random.shuffle(site2)
random.shuffle(site3)


train_list_1 = site1[:n_train_1]
train_list_2 = site2[:n_train_2]
train_list_3 = site3[:n_train_3]

val_list_1 = site1[n_train_1:(n_train_1+n_val_1)]
val_list_2 = site2[n_train_2:(n_train_2+n_val_2)]
val_list_3 = site3[n_train_3:(n_train_3+n_val_3)]

test_list_1 = site1[(n_train_1+n_val_1):]
test_list_2 = site2[(n_train_2+n_val_2):]
test_list_3 = site3[(n_train_3+n_val_3):]


train_images = []
val_images = []
test_images = []
train_masks = []
val_masks = []
test_masks = []

for i in range(filenames.shape[0]):
    fileN = int(filenames[i].split('_')[1][3:])

    if(fileN in train_list_1 or fileN in train_list_2 or fileN in train_list_3):
        train_images.append(images[i])
        train_masks.append(masks[i])
    elif(fileN in val_list_1 or fileN in val_list_2 or fileN in val_list_3):
        val_images.append(images[i])
        val_masks.append(masks[i])
    elif(fileN in test_list_1 or fileN in test_list_2 or fileN in test_list_3):
        test_images.append(images[i])
        test_masks.append(masks[i])


train_images = np.array(train_images)
val_images = np.array(val_images)
test_images = np.array(test_images)
train_masks = np.array(train_masks)
val_masks = np.array(val_masks)
test_masks = np.array(test_masks)

print("train_images", train_images.shape)
print("val_images", val_images.shape)
print("test_images", test_images.shape)
print("train_mask", train_masks.shape)
print("val_mask", val_masks.shape)
print("test_mask", test_masks.shape)


np.save("multiclass_saved_arrays/" + CATEGORY + "_datasets/" + "_train_images.npy", train_images)
np.save("multiclass_saved_arrays/" + CATEGORY + "_datasets/" + "_val_images.npy", val_images)
np.save("multiclass_saved_arrays/" + CATEGORY + "_datasets/" + "_test_images.npy", test_images)
np.save("multiclass_saved_arrays/" + CATEGORY + "_datasets/" + "_train_masks.npy", train_masks)
np.save("multiclass_saved_arrays/" + CATEGORY + "_datasets/" + "_val_masks.npy", val_masks)
np.save("multiclass_saved_arrays/" + CATEGORY + "_datasets/" + "_test_masks.npy", test_masks)






