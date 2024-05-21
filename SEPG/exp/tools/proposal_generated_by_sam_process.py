from heapq import merge
import json
import os
import argparse
from pycocotools.coco import COCO
from mmdet.core.bbox import bbox_overlaps
from tqdm import tqdm
import torch

'''
python exp/tools/proposal_generated_by_sam_process.py 
    --proposal-dir work-dir/COCO/SAM_PRNetv7_Headv11_MSEv7_lr0.02_thr_0.7_2iter/2iter/SAM_predict_proposals/ 
    --ori-pointann data/COCO/annotations/instances_train2017_coarse.json 
    --ori-ann data/COCO/annotations/instances_train2017.json 
    c work-dir/COCO/SAM_PRNetv7_Headv11_MSEv7_lr0.02_thr_0.7_2iter/2iter/SAM_predict_proposals/
'''


def merge_json(proposal_dir):
    dir_list = os.listdir(proposal_dir)
    dir_list.sort(key= lambda x: int(x.split('start')[1].split('_')[0]))
    merge_list = []

    for file in dir_list:
        print(file)
        with open(os.path.join(proposal_dir, file), 'r') as f:
            pro = json.load(f)
            merge_list.extend(pro)
    
    print('writing results!')
    with open(os.path.join(proposal_dir, 'all_proposals.json'), 'w') as f:
        json.dump(merge_list, f)
    print('end!')

def result2ann(args):
    det_file = args.det_file if args.det_file else os.path.join(args.proposal_dir, 'all_proposals.json')
    coco = COCO(args.ori_ann)
    
    # anns = coco.getAnnIds()
    print(len(coco.getAnnIds()))       # the number of annotations in ori_ann (860001)
    
    res = coco.loadRes(det_file)
    print(len(res.getAnnIds()))        # the number of annotations in results (849947)
    
    iou_sum = 0
    num_sum = 0
    iou_list = []
    corloc = [0,0,0,0,0]
    thr_list = [0.5, 0.6, 0.7, 0.8, 0.9]
    for im_id in tqdm(coco.imgToAnns):
        if im_id in res.imgToAnns:
            anns = res.imgToAnns[im_id]
            assert len(anns) != 0
            ann_delete_list = []
            for ann in anns:
                ori_ann = coco.loadAnns(ann['ann_id'])[0]
                
                if ori_ann['iscrowd'] == True:
                    continue
                assert ori_ann['id'] == ann['ann_id'], f"{ori_ann} vs {ann}"
                for key in ['image_id', 'category_id', 'iscrowd']:
                    assert ori_ann[key] == ann[key], key
                
                for key in ['bbox', 'segmentation', 'area', ]:
                    if key == 'bbox':
                        #                         print(torch.tensor(ori_ann[key]).unsqueeze(-1).shape)
                        ba = torch.tensor(ori_ann[key]).unsqueeze(0)
                        ba[:, 2:4] = ba[:, 0:2] + ba[:, 2:4]
                        bb = torch.tensor(ann['bbox']).unsqueeze(0)
                        bb[:, 2:4] = bb[:, 0:2] + bb[:, 2:4]
                        iou = bbox_overlaps(ba, bb)
                        iou_list.append(iou.item())
                        iou_sum += iou
                        num_sum += 1
                        ori_ann['iou'] = iou.item()
                        
                        ori_ann[key] = ann['bbox']
                    else:
                        ori_ann[key] = ann[key]
                ## add by fei
                # ori_ann['ann_weight'] = ann['score']
                
                # delete annotaions iou lower than 0.5
                # added by guo
                for i in range(len(thr_list)):
                    if iou > thr_list[i]:
                        corloc[i] +=1

            # ann_delete_list.reverse()
            # for ann_delete_num in ann_delete_list:
            #     anns.pop(ann_delete_num)
            
    print(len(ann_delete_list))
    print(num_sum)

    corloc = [corloc[i] / num_sum for i in range(len(corloc))]
    mean_iou = iou_sum / num_sum
    
    #added by guo
    # iou_large_than_05 = [iou for iou in iou_list if iou > 0.5]
    # print('the number of large iou:',len(iou_large_than_05))
    # json.dump(iou_list, open('/home/ubuntu/Guo/P2BNet-main/TOV_mmdetection/work-dir/coco/iou.json', 'w'))
    # json.dump(iou_large_than_05, open('/home/ubuntu/Guo/P2BNet-main/TOV_mmdetection/work-dir/coco/iou_large_than_05.json', 'w'))
    
    
    print('mean_iou:', mean_iou, 'num_sum:', num_sum, 'corloc:', corloc)

    # check(coco, res)
    print('saving the results!')
    save_ann = os.path.join(args.save_dir, 'coco_latest_results_all.json')
    json.dump(coco.dataset, open(save_ann, 'w'))
    print('end!')


def filt_error_seg(args):
    json_path = os.path.join(args.save_dir, 'coco_latest_results_all.json')
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
                # print(ann, img_id)
                num+=1
                
                ####### Good method! ###############
                # x,y,w,h = ann['bbox']
                # x1,y1 = x+w, y
                # x2, y2 = x+w, y+h
                # x3, y3 = x, y+h
                # x4, y4 = x, y
                # ann['segmentation'] = [[x1, y1, x2, y2, x3, y3, x4, y4]]
                
                ############# Bad method! ################
                coco.dataset['annotations'].remove(ann)
            
            # if len(ann['segmentation'][0]) % 2 != 0:
            #     print(ann, img_id)
    print(num)
    
    print(len(coco.dataset['annotations']))
    with open( os.path.join(args.save_dir, 'results_with_iou_filter.json'), 'w') as f:
        json.dump(coco.dataset, f)


def replace_gt_by_proposals(args):
    # load coco dataset
    coco = COCO(args.ori_pointann)
    # res = coco.loadRes(args.proposal_path)
    proposal_path = os.path.join(args.save_dir, 'coco_latest_results_all.json')
    proposals = COCO(proposal_path)
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
    print('saving the result!')
    with open(os.path.join(args.save_dir, 'instances_train2017_coarse_proposal_adjcent_2iter.json'), 'w') as f:
        json.dump(coco.dataset, f)
    print('end!')

# with open(os.path.join(proposal_dir, 'all_proposals.json'), 'r') as f:
#     merge_dict = json.load(f)

# print(len(merge_dict))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-dir", type=str, default='')
    parser.add_argument("--ori-pointann", help='such as exp/rr_latest_result.json')
    parser.add_argument("--ori-ann", help='such as data/coco/resize/annotations/instances_val2017_100x167.json', default='data/COCO/annotations/instances_train2017.json')
    parser.add_argument("--det-file", help='such as exp/latest_result.json',default=None)
    parser.add_argument("--save-dir", help='such as exp/rr_latest_result.json')
    
    args = parser.parse_args()
    print('============= begin merging! ===============')
    merge_json(args.proposal_dir)
    print('============= transform SAM predictions to cocofmt ann! ==========')
    result2ann(args)
    print('============= filt err seg! ==========')
    filt_error_seg(args)
    
    print('=============replace pse boxes in point ann to sam predicted proposals=========')
    replace_gt_by_proposals(args)
    
    