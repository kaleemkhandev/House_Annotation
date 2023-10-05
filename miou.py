import os
import shutil
import glob
import cv2
import numpy as np


def calculate_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)
    if np.sum(union)==0:
        return 0
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_miou(gt_masks, pred_masks, num_classes):
    miou_sum = 0.0
    num_images = len(gt_masks)
    for i in range(num_images):
        gt_image_masks = gt_masks[i]
        pred_image_masks = pred_masks[i]
        iou_sum = 0.0
        for class_id in range(num_classes):
            gt_class_mask = (gt_image_masks == class_id).astype(np.uint8)
            pred_class_mask = (pred_image_masks == class_id).astype(np.uint8)
            iou = calculate_iou(gt_class_mask, pred_class_mask)
            iou_sum += iou
        miou_sum += iou_sum / len(list(np.unique(gt_image_masks)))
    miou = miou_sum / num_images
    return miou



gt_masks = []
pred_masks = []
files = os.listdir('seg_data/only_seg/images/')
for file in files:
    fl_pred = file.replace('.jpg','_pred.png')
    fl_gnd = file.replace(".jpg","_gt.png")
    label_file = f"seg_data/only_seg/gnd_truth/{fl_gnd}"
    pred_file = f"seg_data/only_seg/pred/{fl_pred}"
    label_mask = cv2.imread(label_file,cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.imread(pred_file,cv2.IMREAD_GRAYSCALE)
    # pred_mask[pred_mask == 1] = 4
    # # Replace twos with ones
    # pred_mask[pred_mask == 2] = 1
    # # Replace threes (originally ones) with twos
    # pred_mask[pred_mask == 3] = 2
    # # Replace temporary value 4 with threes
    # pred_mask[pred_mask == 4] = 3
    
    gt_masks.append(label_mask)
    pred_masks.append(pred_mask)
    # break


num_classes = 4

miou = calculate_miou(gt_masks, pred_masks, num_classes)
print("mIoU:", miou)