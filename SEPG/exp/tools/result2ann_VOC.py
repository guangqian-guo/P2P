from pycocotools.coco import COCO
import json
import argparse
from mmdet.core.bbox import bbox_overlaps
import torch


def check(coco, res):
    for im_id in coco.imgToAnns:
        if im_id in res.imgToAnns:
            anns = res.imgToAnns[im_id]
            for ann in anns:
                ori_ann = coco.loadAnns(ann['ann_id'])[0]

                assert ori_ann['id'] == ann['ann_id']
                for key in ['bbox', 'segmentation', 'area', ]:
                    assert key in ori_ann, ori_ann
                    assert key in ann, ann
                    assert ori_ann[key] == ann[key], f"{key}\n\t{ori_ann}\n\t{ann}"


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
    
    for im_id in coco.imgToAnns:
        if im_id in res.imgToAnns:
            anns = res.imgToAnns[im_id]
            ann_delete_list = []
            for ann in anns:
                ori_ann = coco.loadAnns(ann['ann_id'])[0]
                assert ori_ann['id'] == ann['ann_id'], f"{ori_ann} vs {ann}"

                for key in ['image_id', 'category_id']:
                    assert ori_ann[key] == ann[key], key

                for key in ['bbox', 'segmentation', 'area', ]:
                    if key == 'bbox':
                        #                         print(torch.tensor(ori_ann[key]).unsqueeze(-1).shape)
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
                # ori_ann['ann_weight'] = ann['score']
                
                # delete annotaions iou lower than 0.5
                # added by guo
                if iou < 0.5:
                    ann_delete_list.append(ann['ann_id'])

            # ann_delete_list.reverse()
            # for ann_delete_num in ann_delete_list:
            #     anns.pop(ann_delete_num)
            
    print(len(ann_delete_list))
    


    mean_iou = iou_sum / num_sum
    
    #added by guo
    # iou_large_than_05 = [iou for iou in iou_list if iou > 0.5]
    # print('the number of large iou:',len(iou_large_than_05))
    # json.dump(iou_list, open('/home/ubuntu/Guo/P2BNet-main/TOV_mmdetection/work-dir/coco/iou.json', 'w'))
    # json.dump(iou_large_than_05, open('/home/ubuntu/Guo/P2BNet-main/TOV_mmdetection/work-dir/coco/iou_large_than_05.json', 'w'))
    

    print('mean_iou:', mean_iou, 'num_sum:', num_sum)

    check(coco, res)
    json.dump(coco.dataset, open(args.save_ann, 'w'))