"""
Functions to evaluate semantic segmentation models
"""

import numpy as np



# intersection over : percent overlap between the target mask and our prediction
def mean_iou_test(y_true, y_pred, num_classes):
    intersection = np.zeros(num_classes) # int = (mesh : M, wire : W and background : B)
    den = np.zeros(num_classes) # den = M + W + B = (M or W or B) + (M and W and B) + ..

    for i in range (len(y_true)):
        for j in range(512):
            for k in range(512):
                if y_pred[i][j][k]==y_true[i][j][k]:
                    intersection[y_true[i][j][k]]+=1
                den[y_pred[i][j][k]] += 1
                den[y_true[i][j][k]] += 1
    mIoU = 0
    for i in range(num_classes):
        if den[i]!=0:
            mIoU+=intersection[i]/(den[i]-intersection[i])
        else:
            mIoU+=1
    mIoU=mIoU/num_classes
    return mIoU






# dice coefficient
def dice_coeff(y_true, y_pred, num_classes):
    intersection = np.zeros(num_classes) # int = (mesh : M, wire : W and background : B)
    den = np.zeros(num_classes) # den = M + W + B = (M or W or B) + (M and W and B) + ..

    for i in range (len(y_true)):
        for j in range(512):
            for k in range(512):
                if y_pred[i][j][k]==y_true[i][j][k]:
                    intersection[y_true[i][j][k]]+=1
                den[y_pred[i][j][k]] += 1
                den[y_true[i][j][k]] += 1
    dice = 0
    for i in range(num_classes):
        if den[i]!=0:
            dice+=2*intersection[i]/2*intersection[i]+(den[i]-intersection[i])
        else:
            dice+=1
    dice=dice/num_classes
    return dice





def pixel_accuracy(y_true, y_pred, num_classes):
    intersection = np.zeros(num_classes) # int = (mesh : M, wire : W and background : B)
    den = np.zeros(num_classes) # den = M + W + B = (M or W or B) + (M and W and B) + ..

    for i in range (len(y_true)):
        for j in range(512):
            for k in range(512):
                if y_pred[i][j][k]==y_true[i][j][k]:
                    intersection[y_true[i][j][k]]+=1
    pix_acc = 0
    for i in range(num_classes):
        if den[i]!=0:
            pix_acc+=intersection[i]
        else:
            pix_acc+=1
    pix_acc=pix_acc/num_classes
    return pix_acc


def pixel_accuracy_multiclass(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = (y_pred[:,:,:,0] > 0.5).astype(int)
    intersection = np.logical_and(yt0, yp0) #TP
    class1 = np.sum(yt0) #TP+FN
    acc = np.sum(intersection)/class1
    return acc