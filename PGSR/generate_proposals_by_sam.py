#--------------------------------------
# Generate Proposals by SAM
# Author: guo guangqian
# Data: 2023/7/12   use point and pred bbox as prompt, generate refined bbox and refined masks 
#--------------------------------------



from cProfile import label
from ctypes import pointer
from tkinter import image_names
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
# import pydicom
from sam.segment_anything import SamPredictor, sam_model_registry, SamPredictorPreextract
import os
import json
from collections import defaultdict
import argparse
from pycocotools.coco import COCO
from tqdm import tqdm
from simplification.cutil import simplify_coords_vwp

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 200/255, 0/255, 0.6])
    
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)   

def show_masks(masks, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 200/255, 0/255, 0.6])
    for mask in masks: 
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image) 
        
def show_box(box, ax, color='green'):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2))    

def show_boxes(boxes, ax, color='green'):
    for box in boxes:
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2))    

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25) 

def show_pointsv2(coords, ax, marker_size=200):
    pos_points = coords
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)



def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # load coco dataset
    coco = COCO(args.point_annpath)
    pse_coco = COCO(args.pse_path)
    
    num_imgs = len(coco.getImgIds())
    print(f'There are {num_imgs} images.')

    # load SAM
    checkpoint_registry = {
        "vit_b": "sam_vit_b_01ec64.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_t": "MobileSAM-master/weights/mobile_sam.pt",
    }
    
    sam = sam_model_registry[args.model](checkpoint=checkpoint_registry[args.model])
    sam.to(device='cuda')
    predictor = SamPredictorPreextract(sam)
    
    empty_ann = 0
    
    # predict masks
    pred_proposals = []
    if args.end == -1:
        end = num_imgs
    else:
        end = args.end
    
    idx_end = args.start
    idx_start = args.start
    for img_id in tqdm(coco.getImgIds()[args.start: end]):
        idx_end += 1
        # if img_id == 484200:
        #     print(idx_end)
        #     exit()
        # continue
        img_info = coco.loadImgs(img_id)[0]
        img_name = img_info['file_name']

        img_path = os.path.join(args.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        
        # feat = np.load(os.path.join(args.feat_path, img_name.split('.')[0]+'.npy'))
        # predictor.features = torch.from_numpy(feat).cuda()

        ann_ids = coco.getAnnIds(img_id)
        anns = coco.loadAnns(ann_ids)
        if len(anns) == 0:
            empty_ann += 1
            continue
    
        pred_proposals_per_img = []
        point_gt_per_img = []
        for ann in anns:
            ann_id = ann['id']
            pse_ann = pse_coco.loadAnns(ann_id)[0]
            pred_bbox = np.array(pse_ann['bbox'])
            pred_bbox[2:] = pred_bbox[:2] + pred_bbox[2:]  # xywh-->xyxy
            category_id = ann['category_id']
            point_gt_per_img.append(ann['point'])
            point = np.array(ann['point'])
            label = np.array([1])
            masks, scores, _ = predictor.predict(point_coords=np.expand_dims(point,axis=0),
                                            point_labels=label,
                                            box=pred_bbox[None],
                                            multimask_output=False)
            # masks, scores, _ = predictor.predict(point_coords=None,
            #                                 point_labels=label,
            #                                 box=pred_bbox[None],
            #                                 multimask_output=False)
            
            mask = masks[0]
            # plt.figure(figsize=(10, 10))
            # plt.imshow(image)
            # show_mask(mask, plt.gca())
            # plt.show()
            grey_map = (mask.astype(int) * 255).astype(np.uint8)
            _, thr_map = cv2.threshold(grey_map, 1, 255, cv2.THRESH_TOZERO)
            # np.set_printoptions(threshold=np.inf)
            contours, _ = cv2.findContours(thr_map, 
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) != 0:
                c = max(contours, key=cv2.contourArea)
                
                all_contour = []
                for contour in contours:
                    all_contour.extend(contour)
                c_ = cv2.convexHull(np.array(all_contour))
                bbox = list(cv2.boundingRect(c_))
                # bbox = list(cv2.boundingRect(c))
                sc = simplify_coords_vwp(c[:,0,:], 2).ravel().tolist()
                if sc[0] < 1e-5 and sc[1]<1e-5:
                    sc = sc[2:] 
                
                ann_dict = {'image_id': img_id, 'bbox':bbox, 'segmentation':[sc], 'ann_id': ann_id, 'category_id': category_id}
                pred_proposals_per_img.append(ann_dict)
            else:
                print(img_name)
        
        pred_proposals.extend(pred_proposals_per_img)
        
        # save prooposals
        if idx_end >= 2000 and idx_end % 2000 == 0:
            with open(os.path.join(args.save_dir, 'proposals_start%d_end%d.json' % (idx_start, idx_end)), 'w') as f:
                json.dump(pred_proposals, f)
            pred_proposals = []
            idx_start = idx_end
    
    # save the last results
    with open(os.path.join(args.save_dir, 'proposals_start%d_end%d.json' % (idx_start, idx_end)), 'w') as f:
        json.dump(pred_proposals, f, indent=4)
    print(idx_end)
    
        # show 
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # show_mask(mask, plt.gca())
        # try:
        #     show_pointsv2(np.array(point_gt_per_img), plt.gca())
        # except:
        #     print(point_gt_per_img)
        # show_boxes(pred_proposals_per_img, plt.gca(), color='red')
        # plt.axis('off')
        # plt.show()   # show img
        # save fig
        # plt.savefig(args.save_dir+img_name)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", help='such as data/coco/resize/annotations/instances_val2017_100x167.json')
    parser.add_argument("--model", default="vit_h", help='such as data/coco/resize/annotations/instances_val2017_100x167.json')
    parser.add_argument("--point-annpath", help='such as exp/rr_latest_result.json')
    parser.add_argument("--pse-path", help='such as exp/rr_latest_result.json')
    parser.add_argument("--save-dir")
    parser.add_argument("--feat_path")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    args = parser.parse_args()
    
    main(args)

