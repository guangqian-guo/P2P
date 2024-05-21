import cv2
import random
import json, os
from pycocotools.coco import COCO
from skimage import io
from matplotlib import pyplot as plt

train_json = '/home/ps/Guo/P2BNet-main/TOV_mmdetection/work-dir/COCO/SAM_PRNetv6_Headv4_lr0.02/SAM_refine_proposals/results.json'
train_json = '/home/ps/Guo/P2BNet-main/TOV_mmdetection/work-dir/COCO/SAM_PRNetv6_Headv4_lr0.02/SAM_refine_proposals/results_with_iou.json'
# train_json = '/home/ps/Guo/P2BNet-main/TOV_mmdetection/data/COCO/annotations/instances_train2017.json'
train_path = '/home/ps/Guo/P2BNet-main/TOV_mmdetection/data/COCO/train2017/'
train_json = '/home/ps/Guo/P2BNet-main/TOV_mmdetection/work-dir/COCO/SAM_PRNetv3_lr0.02v3/SAM_refine_proposal/results_with_iou_filter.json'
train_json = '/home/ps/Guo/P2BNet-main/TOV_mmdetection/work-dir/COCO/SAM_PRNetv5_Headv6_lr0.02/SAM_refine_proposal/results_with_iou.json'
train_json = '/home/ps/Guo/P2BNet-main/TOV_mmdetection/work-dir/COCO/SAM_PRNetv5_Headv8_lr0.02v2/SAM_refine_proposals/results_with_iou.json'
train_json = '/home/ps/Guo/P2BNet-main/TOV_mmdetection/work-dir/COCO/SAM_PRNetv7_Headv11_lr0.02v3/SAM_refined_proposals/results_with_iou.json'
def visualization_seg(img_id, json_path, img_path):
    # 需要画图的是第num副图片, 对应的json路径和图片路径,
    # str = ' '为类别字符串，输入必须为字符串形式 'str'，若为空，则返回所有类别id
    coco = COCO(json_path)
    img = coco.loadImgs(img_id)[0]  # 加载图片,loadImgs() 返回的是只有一个内嵌字典元素的list, 使用[0]来访问这个元素
    print(img)
    image = io.imread(img_path + img['file_name'])
    
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    print(anns)
    # 读取在线图片的方法
    # I = io.imread(img['coco_url'])
    
    plt.imshow(image) 
    coco.showAnns(anns)
    plt.show() 

def filt_error_seg(json_path):
    coco = COCO(json_path)
    print(len(coco.dataset['annotations']))
    
    num = 0
    for img_id in coco.imgToAnns:
        ann_ids = coco.getAnnIds(img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if ann['iscrowd']:
                continue
            if len(ann['segmentation'][0]) < 8:
                print(ann, img_id)
                num+=1
                x,y,w,h = ann['bbox']
                x1,y1 = x+w, y
                x2, y2 = x+w, y+h
                x3, y3 = x, y+h
                x4, y4 = x, y
                ann['segmentation'] = [[x1, y1, x2, y2, x3, y3, x4, y4]]
                # coco.dataset['annotations'].remove(ann)
            # if len(ann['segmentation'][0]) % 2 != 0:
            #     print(ann, img_id)
    print(num)
    
    print(len(coco.dataset['annotations']))
    with open('/home/ps/Guo/P2BNet-main/TOV_mmdetection/work-dir/COCO/SAM_PRNetv7_Headv11_lr0.02v3/SAM_refined_proposals/results_with_iou_filter.json', 'w') as f:
        json.dump(coco.dataset, f)
        
if __name__ == "__main__":
#    visualization_seg(433052, train_json, train_path)
   filt_error_seg(train_json)


