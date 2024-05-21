# Author: Guo Guangqian
# Data: 2023/6/25 
# youhua
# v2 version  outputs pse bboxes that larger or lower than a thr

import json
from collections import defaultdict
import os
from random import weibullvariate
import random
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
import argparse
from  pycocotools.coco import COCO
from tqdm import tqdm
from random import sample
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
    root = args.img_root
    pse_gt_path = args.pse_gt_path
    gt_path = args.gt_path
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    num = args.num
    pse_gt = COCO(pse_gt_path)
    if gt_path is not None:
        gt = COCO(gt_path)
    
    random_id = sample(pse_gt.getImgIds(), num)
    for img_id in tqdm(random_id):
        img_info = pse_gt.loadImgs(img_id)[0]
        img_name = img_info['file_name']
        img_path = os.path.join(root, img_name)
        pse_anns_ids = pse_gt.getAnnIds(img_id)
        pse_anns = pse_gt.loadAnns(pse_anns_ids)
        
        plt.figure(dpi=250)
        ax = plt.gca()
        ax.axis('off')

        img = Image.open(img_path)
        img = np.array(img)

        color = (255, 140, 0)
        color_1 = (1, 140 / 255, 0)
        flag = 0
        for pse_ann in pse_anns:
            if pse_ann['iscrowd']:
                continue
            try:
                iou = pse_ann['iou']
                # score = pse_ann['ann_weight']
            except:
                print(pse_ann)
            if iou < args.thr:
                flag = 1
                bb = pse_ann['bbox']
                class_name = coco_id_name_map[pse_ann['category_id']]
                bb = [int(x) for x in bb]
                top_left, bottom_right = bb[:2], [bb[0] + bb[2], bb[1] + bb[3]]
                img = cv2.rectangle(img, tuple(top_left), tuple(bottom_right), (0,229,238),
                                    1)  # blue(0,229,238) yellow(255,215,0) green(127,255,0) orange(255,140,0)
                label_text = class_name + '|' + f'{iou:.02f}'
                # label_text = f'{score:.2f}' + '|' + f'{iou:.02f}'
                # add text
                ax.text(
                    bb[0],
                    bb[1],
                    f'{label_text}',
                    bbox={
                        'facecolor': 'black',
                        'alpha': 0.8,
                        'pad': 0.7,
                        'edgecolor': 'none'
                    },
                    color=color_1,
                    fontsize=8,
                    verticalalignment='top',
                    horizontalalignment='left')

        # 保存图片
        if flag:
            flag=0
            plt.imshow(img)
            plt.savefig(os.path.join(save_dir, img_name), bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-root", type=str, default='data/COCO/train2017/')
    parser.add_argument("--pse-gt-path", type=str)
    parser.add_argument("--gt-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--thr", type=float, default=0.3 )
    args = parser.parse_args()

    main(args)
