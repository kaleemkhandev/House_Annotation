import os
import shutil
import glob
import cv2
import numpy as np


def calculate_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou




miou_sum = 0.0
files = os.listdir('seg_data/images/')
for file in files:
    fl_pred = file.replace('.jpg','_pred.png')
    fl_gnd = file.replace(".jpg","_gt.png")
    label_file = f"seg_data/gnd_truths/{fl_gnd}"
    pred_file = f"seg_data/pred/{fl_pred}"
    label_mask = cv2.imread(label_file,cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.imread(pred_file,cv2.IMREAD_GRAYSCALE)
    num_classes = 4

    iou_sum = 0.0
    # cv2.imshow("gd_truth",label_mask)
    # cv2.waitKey(0)
    # cv2.imshow("pred",label_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    for class_id in range(num_classes):
        gt_class_mask = (label_mask == np.uint8(class_id)).astype(np.uint8)
        pred_class_mask = (pred_mask == np.uint8(class_id)).astype(np.uint8)
        intersection = np.logical_and(gt_class_mask, pred_class_mask)
        union = np.logical_or(gt_class_mask, pred_class_mask)
        iou = np.sum(intersection) / np.sum(union)
        if np.sum(union) == 0:
            print(iou)
            print('hey')
        iou_sum += iou

    iou_for_one = iou_sum / num_classes
    miou_sum+=iou_for_one

miou = miou_sum/len(files)
print("mIoU:", miou)



import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
import glob
from segment_anything import sam_model_registry, SamPredictor

def show_anns(anns,color_map:dict):
    if len(anns) == 0:
        return

    img = np.ones((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for i,ann in enumerate(anns):
        m = ann['segmentation']
        color_mask = color_map[str(i)]
        img[m] = color_mask

    return img

def click_handler(cord,masks):
    hist = []
    counter = 0
    for i, mask in enumerate(masks):
        if mask['segmentation'][cord[0][1]][cord[0][0]]:
            # print(i)
            src_clr = color_map[str(i)]
            # break
        
    for i,mask in enumerate(masks):
        if mask['segmentation'][cord[1][1]][cord[1][0]]:
            counter+=1
            hist.append(i)
            # print(i)
    if len(hist)<2:
        try:
            print(hist)
            color_map[str(hist[0])] = src_clr
        except Exception as e:
            print(e, '(Please Click on Segmented Region.)') 
    else:
        indx = find_smallest(hist,masks,cord[1])
        color_map[str(indx)] = src_clr
        
    return color_map

def find_smallest(hist,masks,cord):
    for indx in reversed(hist):
        if masks[indx]['segmentation'][cord[1]][cord[0]]:
            return indx

def click_event(event, x, y, flags, param):
    global coords
    global final
    if event == cv2.EVENT_LBUTTONDOWN:  # double left click event
        coords.append((x, y))
        if len(coords) == 2:
            print(coords)
            color_map = click_handler(coords, masks)
            masked_img = show_anns(masks,color_map)
            image_test = image_original / 255
            overlay = image_test * 0.4 + masked_img[:, :, 0:3]*0.6
            final = overlay
            # cv2.line(overlay, coords[0], coords[1], (0, 255, 0), 2)
            cv2.imshow('image', overlay)
            coords = []

label_map = {
'0':[0, 0, 1, 0.35],
'1':[0, 1, 0, 0.35],
'2':[1, 0, 0, 0.35]
}
def show_mask(mask, ax=None, classes=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        
    elif classes is not None and random_color is False:
        color = label_map[str(classes)]
        color = np.array(color)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
        
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    # img = np.ones((mask[0].shape[0], mask[0].shape[1], 4))
    # img[:,:,3] = 0
    # for i,ann in enumerate(anns):
    #     m = ann['segmentation']
    #     color_mask = label_map[str(i)]
    #     img[m] = color_mask

    # return mask_image
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


height = 720
width = 1280
files = glob.glob("yolov8_data/gnd_truths/*")
for file in files:
    print(file)
    with open(file,'r') as f:
        lines = f.readlines()
    labels = []
    bboxes = []
    # print(lines)
    for line in lines:
        line = line.strip('\n').split(' ')
        line = [float(i) for i in line]
        labels.append(int(line[0]))
        bboxes.append(line[1:])
    bbox =[]
    for box in bboxes:
        x,y,w,h = int(box[0]*width),int(box[1]*height),int(box[2]*width),int(box[3]*height)
        x1,y1,x2,y2 = x-(w/2),y-(h/2),x+(w/2),y+(h/2)
        # x1,y1,x2,y2 = x,y,x+(w/2),y+(h/2)
        box = [int(x1),int(x2),int(y1),int(y2)]
        bbox.append(box)

    #lets read the image
    ref = file.split('/')[-1]
    input_test_image = 'yolov8_data/images/'+ref.replace('_gt.txt','.jpg')
    image = cv2.imread(input_test_image)
    image = cv2.resize(image, (1280, 720))
    for box in bbox:
        image = cv2.rectangle(image, (box[0],box[2]), (box[1],box[3]), (255, 0, 0), 2)
    cv2.imshow('check',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    image_detect = cv2.imread(input_test_image) 
    image_detect = cv2.resize(image_detect, (1280, 720))
    image_detect = cv2.cvtColor(image_detect, cv2.COLOR_BGR2RGB)

    #lets load the weights for the dam model
    sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cpu"  # Choose what to use for infrerence ('cpu', 'cuda'), CPU recommended if GPU not available


    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.set_image(image_detect)
    input_boxes = torch.tensor(bbox, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image_detect.shape[:2])
    masks_det, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    print('     Seg Done.')
    size = masks_det[0][0].numpy().shape
    mask=np.zeros(size, dtype=np.int8)
    all_masks = np.squeeze(masks_det.numpy().astype(int),axis = 1)
    for i ,each_mask in enumerate(all_masks):
            mask += each_mask * (labels[i]+1)
            

    pred_file = ref.split('_')[0]
    cv2.imwrite(f"seg_data/only_seg/pred/{pred_file}_pred.png",mask)
    binary_mask = np.zeros(size, dtype=np.uint8)
    img_height , img_width = size
    # Iterate over each bounding box and class label
    for bounding_box, class_label in zip(bbox, labels):
        x1 = int(bounding_box[0])       
        y1 = int(bounding_box[1])
        x2 =  int(bounding_box[2])        
        y2 = int(bounding_box[3])

        # Create a binary mask for the current bounding box
        class_mask = np.zeros(size, dtype=np.uint8)
        class_mask[y1:y2,x1:x2] = class_label+1
        # class_mask[y:y+int((height/2)), x:x+int((width/2))] = class_label+1

        # Add the class mask to the binary mask
        binary_mask += class_mask 

    gnd_file = ref.split('_')[0]
    cv2.imwrite(f"seg_data/only_seg/gnd_truth/{gnd_file}_gt.png",binary_mask)
    print('     masks saved.')
        