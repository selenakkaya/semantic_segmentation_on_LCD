import os
import numpy as np
from evaluation_multiclass import mean_iou_test, dice_coeff, pixel_accuracy


src_arr = "saved_arrays_multiclass"


sample_dir = "samples_multiclass" #where to save the predictions
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)



#test for same data and expect result as 1.0 
test_mask_same = np.load(src_arr + "/one_hot_masks.npy")[:5]
pred_masks_same = np.load(src_arr + "/one_hot_masks.npy")[:5]



"""  Quantitative measures"""
print("TEST RESULTS SAME DATA")

test_mask_same=np.argmax(test_mask_same,axis=-1) 
pred_masks_same=np.argmax(pred_masks_same,axis=-1)


dice = dice_coeff(test_mask_same, pred_masks_same,3)
print("dice coeff.", dice)


iou = mean_iou_test(test_mask_same, pred_masks_same, 3)
print("mean iou", iou)


acc = pixel_accuracy(test_mask_same, pred_masks_same,3)
print("pixel acc.", acc)



#test for two different data and expect result different than 1.0 
test_mask_dif = np.load(src_arr + "/one_hot_masks.npy")[:5]
pred_masks_dif = np.load(src_arr + "/one_hot_masks.npy")[6:11]

"""  Quantitative measures"""
print("TEST RESULTS DIFFERENT DATA")

test_mask_dif=np.argmax(test_mask_dif,axis=-1) 
pred_masks_dif=np.argmax(pred_masks_dif,axis=-1)


dice = dice_coeff(test_mask_dif, pred_masks_dif,3)
print("dice coeff.", dice)


iou = mean_iou_test(test_mask_dif, pred_masks_dif, 3)
print("mean iou", iou)


acc = pixel_accuracy(test_mask_dif, pred_masks_dif,3)
print("pixel acc.", acc)