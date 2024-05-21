# --------------------------------------------------------
# dataset split
# Written by Guo Guangqian
# data: 2023/1/12
# --------------------------------------------------------

from ast import arg
from pycocotools.coco import COCO
import json
import argparse
from mmdet.core.bbox import bbox_overlaps
import torch
import tqdm
import mmcv
import random

def data_split(imgs, num):
    sampled_imgs = random.sample(imgs, num)
    sampled_imgs_ids = []
    for sampled_img in sampled_imgs:
        sampled_imgs_ids.append(sampled_img['id'])
    return sampled_imgs, sampled_imgs_ids
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ori_ann", help='such as data/coco/resize/annotations/instances_val2017_100x167.json')
    parser.add_argument("save_ann", help='such as exp/rr_latest_result.json')
    parser.add_argument("sample_ratio", type=float, default=0.2)
    args = parser.parse_args()

    coco = COCO(args.ori_ann)
    
    print(len(coco.getImgIds()))    

    dict_length = len(coco.getImgIds())

    reserved_length = int(dict_length * args.sample_ratio)
    
    reserved_im, reserved_im_id = data_split(coco.dataset['images'], reserved_length)

    print('#------------------------------------------------#')
    print('generate reserved id OVER!')
    print('the number of reserved img: ', len(reserved_im))
    print('#------------------------------------------------#')

    #generate new json
    print('generate new json BEGIN!')
    old_json = coco.dataset
    new_json = {}
    new_json['images'] = reserved_im
    new_json['licenses'] = old_json['licenses']
    new_json['info'] = old_json['info']
    new_json['categories'] = old_json['categories']

    new_annotations = []
    length = len(old_json['annotations'])
    prog_bar = mmcv.ProgressBar(length)
    for ann in old_json['annotations']:
        prog_bar.update()
        if ann['image_id'] in reserved_im_id:
            new_annotations.append(ann)
            # reserved_ann_id.remove(ann['id'])
    
    new_json['annotations'] = new_annotations

    json.dump(new_json, open(args.save_ann, 'w'))
    
    print('OVER!')
