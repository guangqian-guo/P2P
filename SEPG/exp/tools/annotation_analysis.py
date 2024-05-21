# --------------------------------------------------------
# analysis
# Written by Guo Guangqian
# data: 2023/1/12
# --------------------------------------------------------


from pycocotools.coco import COCO
import json
import argparse
from mmdet.core.bbox import bbox_overlaps
import torch


def anns_per_img(coco):
    max_num = 0
    more_than_100 = 0
    img_ids = coco.getImgIds()
    print('the number of imgs:', len(img_ids))
    for img_id in img_ids:
        anns = coco.getAnnIds(img_id)
        # print(anns)
        num_ann = len(anns)
        if num_ann > 100:
            more_than_100 += 1
        if num_ann > max_num:
            max_num = num_ann
    print(more_than_100)
    print(max_num)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-path", help='such as data/coco/resize/annotations/instances_val2017_100x167.json')
    args = parser.parse_args()
    coco = COCO(args.ann_path)
    anns_per_img(coco)
    

    print('the number of anns:',len(coco.getAnnIds()))       # the number of annotations in ori_ann (860001)
    print('the number of imgs:', len(coco.getImgIds()))
    
