from heapq import merge
import json
import os
import argparse


# proposal_dir = 'proposalsv2/'
# proposal_dir = 'Visdrone-box/'
# proposal_dir = 'proposals_gen_by_predbox/'
# proposal_dir = 'proposals_gen_by_SAM_PRNet_pred_bbox/'
# proposal_dir = '/home/ps/Guo/P2BNet-main/TOV_mmdetection/work-dir/COCO/SAM_PRNetv3_lr0.02/proposals_by_sam_box/'
# proposal_dir = '/home/ps/Guo/P2BNet-main/TOV_mmdetection/work-dir/COCO/SAM_PRNetv3_lr0.02v3/SAM_refine_proposal'
# proposal_dir = '/home/ps/Guo/P2BNet-main/TOV_mmdetection/work-dir/COCO/SAM_PRNetv3_lr0.02_proposalv2_adjustercenter_large_scale/SAM_refine_proposal'
# proposal_dir = '/home/ps/Guo/P2BNet-main/TOV_mmdetection/work-dir/COCO/SAM_PRNetv5_Headv6_lr0.02/SAM_refine_proposal'
# proposal_dir = '/home/ps/Guo/P2BNet-main/TOV_mmdetection/work-dir/COCO/SAM_PRNetv5_Headv8_lr0.02v2/SAM_refine_proposals'
# proposal_dir = '/home/ps/Guo/P2BNet-main/TOV_mmdetection/work-dir/COCO/SAM_PRNetv7_Headv11_lr0.02v3/SAM_refined_proposals'
# proposal_dir = '/home/ps/Guo/P2BNet-main/TOV_mmdetection/work-dir/COCO/SAM_PRNetv7_Headv11_lr0.02v3/SAM_refined_proposals_4epoch'

def merge_json(proposal_dir):
    dir_list = os.listdir(proposal_dir)
    dir_list.sort(key= lambda x: int(x.split('start')[1].split('_')[0]))
    merge_list = []

    for file in dir_list:
        print(file)
        with open(os.path.join(proposal_dir, file), 'r') as f:
            pro = json.load(f)
            merge_list.extend(pro)

    with open(os.path.join(proposal_dir, 'all_proposals.json'), 'w') as f:
        json.dump(merge_list, f)

# with open(os.path.join(proposal_dir, 'all_proposals.json'), 'r') as f:
#     merge_dict = json.load(f)

# print(len(merge_dict))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("proposal-dir", type=str, default='')
    
    args = parser.parse_args()
    merge_json(args.proposal_dir)
    
    
    
    