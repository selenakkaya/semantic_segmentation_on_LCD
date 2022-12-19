"""
Functions to evaluate semantic segmentation models
"""

import numpy as np

# intersection over : percent overlap between the target mask and our prediction
def mean_iou_test(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = (y_pred[:,:,:,0] > 0.5).astype(int)
    intersection = np.logical_and(yt0, yp0)
    union = np.logical_or(yt0, yp0)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# dice coefficient
def dice_coeff(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = (y_pred[:,:,:,0] > 0.5).astype(int)
    intersection = np.logical_and(yt0, yp0)
    union = np.logical_or(yt0, yp0)
    dice_coeff = 2*np.sum(intersection)/(np.sum(union) + np.sum(intersection))
    return dice_coeff

def pixel_accuracy(y_true, y_pred):
    t0 = y_true[:,:,:,0]
    yp0 = (y_pred[:,:,:,0] > 0.5).astype(int)
    correct_preds = np.sum(np.equal(t0,yp0))
    tot = y_true.shape[0]*y_true.shape[1]*y_true.shape[2]
    acc = correct_preds/tot
    return acc

