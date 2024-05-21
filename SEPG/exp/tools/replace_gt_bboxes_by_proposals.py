#--------------------------------------
# Generate Proposals by SAM
# Author: guo guangqian
# Data: 2023/6/28
#--------------------------------------

from cProfile import label
from ctypes import pointer
from tkinter import image_names
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2

import os
import json
from collections import defaultdict
import argparse
from pycocotools.coco import COCO
from tqdm import tqdm

def main(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    # load coco dataset
    coco = COCO(args.point_annpath)
    # res = coco.loadRes(args.proposal_path)
    proposals = COCO(args.proposal_path)
    num_imgs = len(coco.getImgIds())
    print(f'There are {num_imgs} images.')

    
    for img_id in tqdm(coco.imgToAnns):
        if img_id in proposals.imgToAnns:
            img_info = proposals.loadImgs(img_id)[0]
            img_w, img_h = img_info['width'], img_info['height']

            anns = proposals.imgToAnns[img_id]
            for ann in anns:
                box = ann['bbox']
                ori_ann = coco.loadAnns(ann['id'])[0]
                point = ori_ann['point']
                x,y,w,h = box
                cx, cy = point
                
                box[0] = max(0, min(cx - w/2., img_w))
                box[1] = max(0, min(cy - h/2., img_h))
                
                ori_ann['bbox'] = box
                
    
    # save the last results
    with open(os.path.join(args.save_dir, 'instances_train2017_coarse_proposal_ac_2iter.json'), 'w') as f:
        json.dump(coco.dataset, f)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-path", help='such as data/coco/resize/annotations/instances_val2017_100x167.json')
    parser.add_argument("--point-annpath", help='such as exp/rr_latest_result.json', default='data/COCO/annotations/instances_train2017_coarse.json')
    parser.add_argument("--save-dir" )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    args = parser.parse_args()
    
    main(args)

