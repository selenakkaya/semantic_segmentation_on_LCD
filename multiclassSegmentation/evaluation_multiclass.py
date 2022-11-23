"""
Functions to evaluate semantic segmentation models
"""

import numpy as np



# correct_pred over : percent overlap between the target mask and our prediction
def mean_iou_test(y_true, y_pred, num_classes):
    correct_pred = np.zeros(num_classes) # int = (mesh : M, wire : W and background : B)
    den = np.zeros(num_classes) # den = M + W + B = (M or W or B) + (M and W and B) + ..

    for i in range (y_true.shape[0]):
        for j in range(y_true.shape[1]):
            for k in range(y_true.shape[2]):
                if (y_pred[i][j][k])==(y_true[i][j][k]):
                    correct_pred[(y_true[i][j][k])]+=1
                den[(y_pred[i][j][k])] += 1
                den[(y_true[i][j][k])] += 1
    mIoU = 0
    for i in range(num_classes):
        if den[i]!=0:
            mIoU+=correct_pred[i]/(den[i]-correct_pred[i])
        else:
            mIoU+=1
    mIoU=mIoU/num_classes
    return mIoU






# dice coefficient
def dice_coeff(y_true, y_pred, num_classes):
    correct_pred = np.zeros(num_classes) # int = (mesh : M, wire : W and background : B)
    den = np.zeros(num_classes) # den = M + W + B = (M or W or B) + (M and W and B) + ..

    for i in range (y_true.shape[0]):
        for j in range(y_true.shape[1]):
            for k in range(y_true.shape[2]):
                if y_pred[i][j][k]==y_true[i][j][k]:
                    correct_pred[y_true[i][j][k]]+=1
                den[y_pred[i][j][k]] += 1
                den[y_true[i][j][k]] += 1
    dice = 0
    for i in range(num_classes):
        if den[i]!=0:
            dice+=2*correct_pred[i]/den[i]
        else:
            dice+=1
    dice=dice/num_classes
    return dice





def pixel_accuracy(y_true, y_pred, num_classes):
    correct_pred = np.zeros(num_classes) # int = (mesh : M, wire : W and background : B)

    for i in range (y_true.shape[0]):
        for j in range(y_true.shape[1]): # height
            for k in range(y_true.shape[2]): # width
                if y_pred[i][j][k]==y_true[i][j][k]:
                    correct_pred[y_true[i][j][k]]+=1
    

    pix_acc = 0
    for i in range(1,num_classes):
        tot = (y_true==i).sum()
        if tot!=0:
            pix_acc+=correct_pred[i]/tot
        else:
            pix_acc+=pix_acc

    pix_acc=pix_acc/(num_classes-1)
    return pix_acc

