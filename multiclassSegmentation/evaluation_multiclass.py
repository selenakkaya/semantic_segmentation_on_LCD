"""
Functions to evaluate semantic segmentation models
"""

import numpy as np



# correct_pred over : percent overlap between the target mask and our prediction
def mean_iou_test(y_true, y_pred, num_classes):
    """
    Return mean IoU for multiclass segmentation, and the list with IoU for each class
    y_true: targets (N,h,w). For each pixel is assigned the id of the class
    y_pred: predictions (N, h,w), where N is the number of examples. For each pixel is assigned the id of the class
    """
   
    return mIoU, IoU_classes






# dice coefficient
def dice_coeff(y_true, y_pred, num_classes):
   """"...
   .
   ..."""
    return mDice, dice_classes





def pixel_accuracy_one_class(y_true, y_pred, num_classes):
    """...
    .
    ..."""
    return pix_acc





def pixel_accuracy(y_true, y_pred, num_classes):
   """.
   ...
   ."""
    return mean_pixel_acc, pix_acc_classes
