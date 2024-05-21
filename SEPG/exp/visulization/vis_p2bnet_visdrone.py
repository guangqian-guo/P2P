

import json
from collections import defaultdict
import os
from random import weibullvariate
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np


coco_id_name_map = {1: 'pedestrian', 2: 'people', 3: 'bicycle', 4: 'car', 5: 'van',
                    6: 'truck', 7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 10: 'motor'
                    }
mode = 2

# 根目录文件
root = 'data/VisDrone/VisDrone2019-DET-train/images_600/'
pse_gt_path = 'work-dir/visdrone/P2B/visdrone_1200_latest_pseudo_ann_1.json'
gt_path = '/home/ubuntu/Guo/P2BNet-main/TOV_mmdetection/data/VisDrone/VisDrone2019-DET-train/coco_fmt_annotations/visdrone2019_train.json'
save_dir = 'visdrone_vis_pse_bbox_p2b/'
num_show = 100
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
# gt_path = '/home/ubuntu/Guo/P2BNet-main/TOV_mmdetection/work-dir/coco/_1200_latest_result.json'
# result_path = "coco/V_16_coco17_quasi_center_point/inference/my_coco_2014_minival/bbox.json"

# with open(result_path,"r") as f:
#     result = json.load(f)

with open(pse_gt_path, "r") as f:
    pse_gt = json.load(f)

with open(gt_path, 'r') as f:
    gt = json.load(f)


# 展现的100个图片的列表
show_img = []
for x in pse_gt['images'][:num_show]:
    show_img.append(x['file_name'])

# 这个函数完全没必要写，因为这里的对应关系很简单
# 把id填充0至12位即可，不需要写循环对应
img2id = defaultdict()
for x in gt['images']:
    if x['file_name'] in show_img:
        img2id[x['file_name']] = x['id']

# 把要展示图片的框位置和标注记录下来
gt_bbox = defaultdict(list)
gt_class = defaultdict(list)
pse_gt_weight = defaultdict(list)
pse_gt_bbox = defaultdict(list)

for x in gt['annotations']:
    if x['image_id'] in img2id.values() and x['iscrowd'] != 1:
        gt_bbox[x['image_id']].append(x['bbox'])
        gt_class[x['image_id']].append(coco_id_name_map[x['category_id']])
        # gt_weight[x['image_id']].append(x['ann_weight'])
        # gt_weight[x['image_id']].append(0)

for x in pse_gt['annotations']:
    if x['image_id'] in img2id.values() and x['iscrowd'] != 1:
        pse_gt_bbox[x['image_id']].append(x['bbox'])
        pse_gt_weight[x['image_id']].append(x['ann_weight'])
# #自己检测结果的可视化的记录数据
# id2bbox = defaultdict(list)
# id2class = defaultdict(list)
# id2score = defaultdict(list)
# score_thr=0.6
# for x in result:
#     if x['image_id'] in img2id.values() and x['score'] > score_thr:
#         id2bbox[x['image_id']].append(x['bbox'])
#         id2class[x['image_id']].append(coco_id_name_map[x['category_id']])
#         id2score[x['image_id']].append(x['score'])


# 展示图片，保存
# 定义一个展示gt或result的变量
gt_or_result = 1

for img_name in show_img[:num_show]:
    img_id = img2id[img_name]
    if gt_or_result == 1:
        bbox = gt_bbox[img_id]
        pse_bbox = pse_gt_bbox[img_id]
        classes = gt_class[img_id]
        weight = pse_gt_weight[img_id]
    else:
        bbox = id2bbox[img_id]
        classes = id2class[img_id]
        scores = id2score[img_id]

    plt.figure(dpi=250)
    ax = plt.gca()
    ax.axis('off')

    img = Image.open(os.path.join(root, img_name))
    img = np.array(img)

    color = (255, 140, 0)
    color_1 = (1, 140 / 255, 0)
    name_out = 'p2bnet'
    # 按照gt和result给予不同的标注
    if gt_or_result == 1:
        # ground truth
        # for i, bb in enumerate(bbox):
        #     bb = [int(x) for x in bb]
        #     top_left, bottom_right = bb[:2], [bb[0] + bb[2], bb[1] + bb[3]]
        #     img = cv2.rectangle(img, tuple(top_left), tuple(bottom_right), color,
        #                         2)  # blue(0,229,238) yellow(255,215,0) green(127,255,0) orange(255,140,0)

            # label_text = classes[i] + '|' + f'{weight[i]:.02f}'

            # # add text
            # ax.text(
            #     bb[0],
            #     bb[1],
            #     f'{label_text}',
            #     bbox={
            #         'facecolor': 'black',
            #         'alpha': 0.8,
            #         'pad': 0.7,
            #         'edgecolor': 'none'
            #     },
            #     color=color_1,
            #     fontsize=8,
            #     verticalalignment='top',
            #     horizontalalignment='left')
        # pse ground truth
        for i, bb in enumerate(pse_bbox):
            bb = [int(x) for x in bb]
            top_left, bottom_right = bb[:2], [bb[0] + bb[2], bb[1] + bb[3]]
            img = cv2.rectangle(img, tuple(top_left), tuple(bottom_right), (0,229,238),
                                1)  # blue(0,229,238) yellow(255,215,0) green(127,255,0) orange(255,140,0)

            label_text = classes[i] + '|' + f'{weight[i]:.02f}'
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

    else:
        for i, bb in enumerate(bbox):
            bb = [int(x) for x in bb]
            top_left, bottom_right = bb[:2], [bb[0] + bb[2], bb[1] + bb[3]]
            img = cv2.rectangle(img, tuple(top_left), tuple(bottom_right), (127, 255, 0), 3)  # (0,229,238)
            cla = classes[i]
            sco = scores[i]
            label_text = cla + '|' + f'{sco:.02f}'
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
                color=(1, 0, 0),
                fontsize=13,
                verticalalignment='top',
                horizontalalignment='left')
    # 保存图片
    plt.imshow(img)
    # if not os.path.exists('vis_pt_bbox_' + name_out):
    #     os.makedirs('vis_pt_bbox_' + name_out)
    # if not os.path.exists('vis_pse_bbox_' + name_out):
    #     os.makedirs('vis_pse_bbox_' + name_out)

    if gt_or_result == 1:
        plt.savefig(os.path.join(save_dir, img_name), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(save_dir, img_name), bbox_inches='tight')
