import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


# Function to calculate IoU between two bounding boxes
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_left = max(x1 - w1 / 2, x2 - w2 / 2)
    y_top = max(y1 - h1 / 2, y2 - h2 / 2)
    x_right = min(x1 + w1 / 2, x2 + w2 / 2)
    y_bottom = min(y1 + h1 / 2, y2 + h2 / 2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou

# Function to calculate precision and recall for a given class
def calculate_precision_recall(det_boxes, gt_boxes, iou_threshold):
    det_boxes = sorted(det_boxes, key=lambda x: x[4], reverse=True)
    tp = np.zeros(len(det_boxes))
    fp = np.zeros(len(det_boxes))
    gt_matched = np.zeros(len(gt_boxes))

    for det_idx, det_box in enumerate(det_boxes):
        det_class = det_box[5]
        det_box = det_box[:4]
        max_iou = 0.0
        max_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_matched[gt_idx] == 1 or gt_box[4] != det_class:
                continue

            iou = calculate_iou(det_box, gt_box[:4])
            if iou > max_iou:
                max_iou = iou
                max_idx = gt_idx

        if max_iou >= iou_threshold and max_idx != -1:
            tp[det_idx] = 1
            gt_matched[max_idx] = 1
        else:
            fp[det_idx] = 1

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recall = tp_cumsum / float(len(gt_boxes))
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + np.finfo(np.float32).eps)

    return precision, recall

# Function to calculate Average Precision (AP) for a given precision-recall curve
def calculate_ap(precision, recall):
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])

    return ap

# Read ground truth annotations from file
# Read ground truth annotations from file
def read_annotations(annot_file, image_width, image_height):
    with open(annot_file, 'r') as f:
        lines = f.read().splitlines()

    annotations = []
    for line in lines:
        values = line.split()
        class_label = int(values[0])
        x_center = float(values[1]) * image_width
        y_center = float(values[2]) * image_height
        width = float(values[3]) * image_width
        height = float(values[4]) * image_height
        annotations.append([x_center, y_center, width, height, class_label])

    return annotations

# Read detected bounding boxes from file
def read_detections(det_file, image_width, image_height):
    with open(det_file, 'r') as f:
        lines = f.read().splitlines()

    detections = []
    for line in lines:
        values = line.split()
        class_label = int(values[0])
        confidence = float(values[1])
        x_center = float(values[2]) * image_width
        y_center = float(values[3]) * image_height
        width = float(values[4]) * image_width
        height = float(values[5]) * image_height
        detections.append([x_center, y_center, width, height, confidence, class_label])

    return detections

# Main function to calculate mAP for multiple images
def calculate_map(images_dir, num_classes, iou_threshold):
    average_precisions = []

    for class_label in range(num_classes):
        gt_boxes_all = []
        det_boxes_all = []

        image_files = os.listdir(images_dir)
        num_images = len(image_files)

        for image_file in image_files:
            if image_file.endswith('.jpg'):
                image_name = os.path.splitext(image_file)[0]
                image_path = os.path.join(images_dir, image_file)
                annot_file = os.path.join(images_dir, image_name + '_gt.txt')
                det_file = os.path.join(images_dir, image_name + '_pred.txt')

                image_width, image_height = get_image_dimensions(image_path)
                gt_boxes = read_annotations(annot_file, image_width, image_height)
                det_boxes = read_detections(det_file, image_width, image_height)

                gt_boxes_all.extend(gt_boxes)
                det_boxes_all.extend(det_boxes)

        precision, recall = calculate_precision_recall(det_boxes_all, gt_boxes_all, iou_threshold)
        ap = calculate_ap(precision, recall)
        average_precisions.append(ap)

    mAP = np.mean(average_precisions)

    return mAP

# Function to get image dimensions
def get_image_dimensions(image_path):
    with Image.open(image_path) as image:
        width, height = image.size
    return width, height

if __name__ == '__main__' :

    # Example usage
    images_dir = 'yolov8_data/resized/combine'  # Directory containing image files and corresponding annotation files
    num_classes = 3  # Number of classes in your dataset
    iou_threshold = 0.4  # IoU threshold for matching detections with ground truth

    mAP = calculate_map(images_dir, num_classes, iou_threshold)
    print('mAP:', mAP)