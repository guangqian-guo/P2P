'''
command:
python exp/tools/adjust_center_points_for_SAM_results.py  --pse-gt-path data/COCO/annotations/instances_train2017_coarse_proposal_v2.json  --save-dir data/COCO/annotations/


'''



import json
from collections import defaultdict
import os
from random import weibullvariate
from types import new_class
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
import argparse
from  pycocotools.coco import COCO
from tqdm import tqdm
from mmdet.core.bbox import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
import torch
coco_id_name_map = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                    77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                    82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                    88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
mode = 2

def main(args):
    # 根目录文件
    pse_gt_path = args.pse_gt_path
    gt_path = args.gt_path
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    pse_gt = COCO(pse_gt_path)
    pse_json = pse_gt.dataset
    print(pse_json.keys())
    
    
    if gt_path is not None:
        gt = COCO(gt_path)
    
    num = 0

    for img_id in tqdm(pse_gt.getImgIds()):
        img_info = pse_gt.loadImgs(img_id)[0]
        img_w, img_h = img_info['width'], img_info['height']

        pse_anns_ids = pse_gt.getAnnIds(img_id)
        pse_anns = pse_gt.loadAnns(pse_anns_ids)

        for pse_ann in pse_anns:
            point = pse_ann['point']
            bbox = pse_ann['bbox']
            x,y,w,h = bbox
            cx, cy = point
            # bbox[0] = cx - w/2.
            # bbox[1] = cy - h/2.

            bbox[0] = max(0, min(cx - w/2., img_w))
            bbox[1] = max(0, min(cy - h/2., img_h))
            if bbox[0] > img_w or bbox[1] > img_h:
                 bbox[0], bbox[1] = x, y
                 num += 1
        
    print(num)


    with open(pse_gt_path.split('/')[-1].split('.')[0]+'_adjust_centers.json', 'w') as f:
        json.dump(pse_gt.dataset, f)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pse-gt-path", type=str)
    parser.add_argument("--gt-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str)
    args = parser.parse_args()
    
    main(args)

