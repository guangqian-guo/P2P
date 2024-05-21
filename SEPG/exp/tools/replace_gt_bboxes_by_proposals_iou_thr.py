#--------------------------------------
# 用SAM 得到的proposal 替换P2BNet标注中的box，bbox是虚拟的box，如果要训练P2B，必须经过这个转换
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


def check(args):
    # load coco dataset
    coco = COCO(args.point_annpath)
    # res = coco.loadRes(args.proposal_path)
    proposals = COCO(args.proposal_path)
    num_imgs = len(proposals.getImgIds())
    print(f'There are {num_imgs} images.')

    for img_id in tqdm(proposals.imgToAnns):
        anns = proposals.imgToAnns[img_id]
        for ann in anns:
            print(ann)
        exit()

def main(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    # load coco dataset
    coco = COCO(args.point_annpath)
    # res = coco.loadRes(args.proposal_path)
    proposals = COCO(args.proposal_path)
    print(len(coco.dataset['annotations']))
    exit()
    num_imgs = len(proposals.getImgIds())
    print(f'There are {num_imgs} images.')

    
    for img_id in tqdm(proposals.imgToAnns):
        anns = proposals.imgToAnns[img_id]
        for ann in anns:
            ori_ann = coco.loadAnns(ann['id'])[0]
            ann['true_bbox'] = ori_ann['true_bbox']
            
    
    # save the last results
    with open(os.path.join(args.save_dir, 'instances_train2017_coarse_proposal_iou@0.7.json'), 'w') as f:
        json.dump(proposals.dataset, f)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-path", help='such as ')
    parser.add_argument("--point-annpath", help='such as exp/rr_latest_result.json')
    parser.add_argument("--save-dir" )
    args = parser.parse_args()
    # check(args)
    main(args)

   