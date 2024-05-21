from ast import arg
from pycocotools.coco import COCO
import json
import argparse
from mmdet.core.bbox import bbox_overlaps
import torch
import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ori_ann", help='such as data/coco/resize/annotations/instances_val2017_100x167.json')
    parser.add_argument("det_file", help='such as exp/latest_result.json')
    parser.add_argument("save_ann", help='such as exp/rr_latest_result.json')
    args = parser.parse_args()

    coco = COCO(args.ori_ann)
    
    # anns = coco.getAnnIds()
    print(len(coco.getAnnIds()))       # the number of annotations in ori_ann (860001)
    
    res = coco.loadRes(args.det_file)
    print(len(res.getAnnIds()))        # the number of annotations in results (849947)

    iou_sum = 0
    num_sum = 0
    iou_list = []
    ann_score_dict = {}
    for im_id in coco.imgToAnns:
        avg_score_per_img = 0
        num_anns_per_img = 0
        if im_id in res.imgToAnns:
            anns = res.imgToAnns[im_id]
            
            for ann in anns:
                ori_ann = coco.loadAnns(ann['ann_id'])[0]
                assert ori_ann['id'] == ann['ann_id'], f"{ori_ann} vs {ann}"

                for key in ['image_id', 'category_id', 'iscrowd']:
                    assert ori_ann[key] == ann[key], key

                for key in ['bbox', 'segmentation', 'area', ]:
                    if key == 'bbox':
                        #print(torch.tensor(ori_ann[key]).unsqueeze(-1).shape)
                        ba = torch.tensor(ori_ann[key]).unsqueeze(0)
                        ba[:, 2:4] = ba[:, 0:2] + ba[:, 2:4]
                        bb = torch.tensor(ann[key]).unsqueeze(0)
                        bb[:, 2:4] = bb[:, 0:2] + bb[:, 2:4]
                        iou = bbox_overlaps(ba, bb)
                        iou_list.append(iou.item())
                        iou_sum += iou
                        num_sum += 1
                    ori_ann[key] = ann[key]
                ## add by fei
                ori_ann['ann_weight'] = ann['score']
                
                # delete annotaions iou lower than 0.5
                # added by guo
                avg_score_per_img += (ann['score'] * iou)
                num_anns_per_img += 1
            avg_score_per_img /= num_anns_per_img    
            

        ann_score_dict[im_id] = avg_score_per_img 

    # selecting ann with score top 50%
    dict_length = len(ann_score_dict)
    # print('the number of reserved anns:', dict_length)
    reserved_length = int(dict_length * 0.2)
    sorted_score_dict = sorted(zip(ann_score_dict.values(), ann_score_dict.keys()),reverse=True)
    reserved_im_id = []
    for score, id in sorted_score_dict[:reserved_length]:
        reserved_im_id.append(id)

    print('#------------------------------------------------#')
    print('generate reserved id OVER!')
    print('the number of reserved img: ', len(reserved_im_id))
    print('#------------------------------------------------#')

    #generate new json
    pseudo_ann = '/home/ubuntu/Guo/P2BNet-main/TOV_mmdetection/work-dir/coco/coco_1200_latest_pseudo_ann_1.json'
    old_json = json.loads(open(pseudo_ann).read())
    new_json = {}
    new_json['images'] = []
    for idx, img in enumerate(old_json['images']):
        if idx % 1000 == 0:
            print(idx/len(reserved_im_id*100), '%')

        if img['id'] in reserved_im_id:
            new_json['images'].append(img)
    
    new_json['licenses'] = old_json['licenses']
    new_json['info'] = old_json['info']
    new_json['categories'] = old_json['categories']

    new_annotations = []
    length = len(old_json['annotations'])
    for i, ann in enumerate(old_json['annotations']):
        if i % 10001 == 0:
            print(i/length*100, '%')
        if ann['image_id'] in reserved_im_id:
            new_annotations.append(ann)
            # reserved_ann_id.remove(ann['id'])
    
    new_json['annotations'] = new_annotations

    json.dump(new_json, open(args.save_ann, 'w'))
    
    print('OVER!')
