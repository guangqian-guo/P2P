# Author: Guo Guangqian
# Data: 2023/6/25 
# output all the pse bbox 
# 
'''
Command:
python exp/visulization/vis_p2bnetv3.py --pse-gt-path ../../segment-anything-main/proposals_with_maskv2/results_with_iou_lower_0.5_inslevel_adjust_centers.json  --save-dir ../../segment-anything-main/proposals_with_maskv2/vis_lower0.7_inslevel_adjsut_centers/  --num 1000
python exp/visulization/vis_p2bnetv3.py --pse-gt-path work-dir/COCO/SAM_PRNetv6_Headv4_lowiou0.5_lr0.02/coco_latest_results_all@0.5.json --save-dir work-dir/COCO/SAM_PRNetv6_Headv4_lowiou0.5_lr0.02/vis/ --num 1000
'''


import json
from collections import defaultdict
import os
from random import weibullvariate
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
import argparse
from  pycocotools.coco import COCO
from tqdm import tqdm
coco_id_name_map = {0: "horse", 1: "person", 2: "bottle", 3: "dog", 4: "tvmonitor", 5: "car", 
                    6: "aeroplane", 7: "bicycle", 8: "boat", 9: "chair", 10: "diningtable",
                    11: "pottedplant", 12: "train", 13: "cat", 14: "sofa", 15: "bird",
                    16: "sheep", 17: "motorbike", 18: "bus", 19: "cow"}
mode = 2

def main(args):
    # 根目录文件
    root = args.img_root
    pse_gt_path = args.pse_gt_path
    gt_path = args.gt_path
    save_dir = os.path.join(args.save_dir + pse_gt_path.split('/')[-1].split('.')[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    num = args.num
    pse_gt = COCO(pse_gt_path)
    if gt_path is not None:
        gt = COCO(gt_path)
    for img_id in tqdm(pse_gt.getImgIds()[:num]):
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

        for pse_ann in pse_anns:
            if pse_ann['iscrowd']:
                continue
            class_name = coco_id_name_map[pse_ann['category_id']]
            try:
                iou = pse_ann['iou'] if 'iou' in pse_ann.keys() else pse_ann['ann_weight']
                label_text = class_name + '|' + f'{iou:.02f}'
            except:
                label_text = class_name
            
            bb = pse_ann['bbox']
            
            bb = [int(x) for x in bb]
            top_left, bottom_right = bb[:2], [bb[0] + bb[2], bb[1] + bb[3]]
            img = cv2.rectangle(img, tuple(top_left), tuple(bottom_right), (255,80, 0),
                                5)  # blue(0,229,238) yellow(255,215,0) green(127,255,0) orange(255,140,0)

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
                color=(1,80/255,0),
                fontsize=8,
                verticalalignment='top',
                horizontalalignment='left')

        # 保存图片
        plt.imshow(img)
        # pse_gt.showAnns(pse_anns, draw_bbox=True)   # show segmentation
        # plt.show()
        plt.savefig(os.path.join(save_dir, img_name), bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-root", type=str, default='data/VOC2007/trainval/JPEGImages/')
    parser.add_argument("--pse-gt-path", type=str)
    parser.add_argument("--gt-path", type=str, default=None)
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--num", type=int, default=100)
    args = parser.parse_args()

    main(args)
