# SAM based progressive refine network
# Author: Guo Guangqian
# Data: 2023/7/3
# HMIL

from cgi import print_directory
import copy
import os
from string import capwords
from xml.dom.minidom import ReadOnlySequentialNamedNodeMap

from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from mmdet.core.bbox import bbox_xyxy_to_cxcywh
from mmdet.core import bbox_cxcywh_to_xyxy
import torch
import numpy as np
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from ..builder import build_head
import os

def bbox_xywh_to_xyxy(bbox):
    x1, y1, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [x1, y1, x1+w, y1+h]
    return torch.cat(bbox_new, dim=-1)


def gen_proposals_from_seed(seed, proposal_cfg, img_meta):
    cut_mode=None
    base_scale = proposal_cfg['base_scales']
    base_ratios = proposal_cfg['base_ratios']
    shake_ratio = proposal_cfg['shake_ratio']
    
    proposal_list = []
    proposals_valid_list = []
    for i in range(len(img_meta)):
        # print(seed[i])
        # print(img_meta[i])
        pps = []
        img_h, img_w, _ = img_meta[i]['img_shape']
        base_boxes = seed[i]
        for scale in base_scale:
            coarse_cluster = []
            # add base proposal
            for ratio in base_ratios:
                base_boxes_ = bbox_xyxy_to_cxcywh(base_boxes)
                base_boxes_[:, 2] *= (ratio * scale)
                base_boxes_[:, 3] *= (ratio * scale)
                base_boxes_ = bbox_cxcywh_to_xyxy(base_boxes_)
                coarse_cluster.append(base_boxes_.unsqueeze(1))
            # print(len(coarse_cluster), coarse_cluster[0].shape)
            coarse_cluster = torch.cat(coarse_cluster, dim=1)
            
            if coarse_cluster.shape[0] == 0:
                print(base_boxes)
                print(img_meta[i])
            
            # add shake proposal   only support 1 shake ratio now TODO!!!
            if shake_ratio is not None:
                for ratio in shake_ratio:
                    coarse_pps = bbox_xyxy_to_cxcywh(coarse_cluster)
                    coarse_pps_center = coarse_pps[:, :, :2]
                    coarse_pps_wh = coarse_pps[:,:,2:4]
                    pps_x_l = coarse_pps_center[:, :, 0] - ratio * coarse_pps_wh[:, :, 0]
                    pps_x_r = coarse_pps_center[:, :, 0] + ratio * coarse_pps_wh[:, :, 0]
                    pps_y_t = coarse_pps_center[:, :, 1] - ratio * coarse_pps_wh[:, :, 1]
                    pps_y_d = coarse_pps_center[:, :, 1] + ratio * coarse_pps_wh[:, :, 1]
                    pps_center_l = torch.stack([pps_x_l, coarse_pps_center[:, :, 1]], dim=-1)
                    pps_center_r = torch.stack([pps_x_r, coarse_pps_center[:, :, 1]], dim=-1)
                    pps_center_t = torch.stack([coarse_pps_center[:, :, 0], pps_y_t], dim=-1)
                    pps_center_d = torch.stack([coarse_pps_center[:, :, 0], pps_y_d], dim=-1)
                    coarse_pps_center = torch.stack([pps_center_l, pps_center_r, pps_center_t, pps_center_d], dim=2)
                    coarse_pps_wh = coarse_pps_wh.unsqueeze(2).expand(coarse_pps_center.shape)
                    coarse_pps = torch.cat([coarse_pps_center, coarse_pps_wh], dim=-1)
                    try:
                        coarse_pps = coarse_pps.reshape(coarse_pps.shape[0], -1, 4)
                    except:
                        print(img_meta[i])
                        exit()
                    coarse_pps = bbox_cxcywh_to_xyxy(coarse_pps)
                    
                    # clamp
                    coarse_pps[..., 0:4:2] = torch.clamp(coarse_pps[..., 0:4:2], 0, img_w-1)
                    coarse_pps[..., 1:4:2] = torch.clamp(coarse_pps[..., 1:4:2], 0, img_h-1)
                    
                    coarse_pps = torch.cat([coarse_cluster, coarse_pps], dim=1)
                    
                    # show_proposals(coarse_pps, img_meta[i])
                    # exit()
                    pps.append(coarse_pps)
            else:
                pps.append(coarse_cluster)
        pps = torch.cat(pps, dim=1)

        # print(pps.shape)
        # show_proposals(pps, img_meta[i])
        proposals_valid = pps.new_full(
            (pps.shape[0], len(base_scale), 1), 1, dtype=torch.long).reshape(-1, 1)  # TODO !!! 3 is num_cluster!!!!
        
        proposals_valid_list.append(proposals_valid)
        proposal_list.append(pps.reshape(-1, 4))
        
        
        
        # pps_old = torch.cat(pps, dim=1)
        # print(pps_old.shape)
    
        # if shake_ratio is not None:
        #     pps_new = []
        #     pps_new.append(pps_old.reshape(*pps_old.shape[0:2], -1, 4))
        #     for ratio in shake_ratio:
        #         pps = bbox_xyxy_to_cxcywh(pps_old)
        #         pps_center = pps[:, :, :2]
        #         pps_wh = pps[:, :, 2:4]
        #         pps_x_l = pps_center[:, :, 0] - ratio * pps_wh[:, :, 0]
        #         pps_x_r = pps_center[:, :, 0] + ratio * pps_wh[:, :, 0]
        #         pps_y_t = pps_center[:, :, 1] - ratio * pps_wh[:, :, 1]
        #         pps_y_d = pps_center[:, :, 1] + ratio * pps_wh[:, :, 1]
        #         pps_center_l = torch.stack([pps_x_l, pps_center[:, :, 1]], dim=-1)
        #         pps_center_r = torch.stack([pps_x_r, pps_center[:, :, 1]], dim=-1)
        #         pps_center_t = torch.stack([pps_center[:, :, 0], pps_y_t], dim=-1)
        #         pps_center_d = torch.stack([pps_center[:, :, 0], pps_y_d], dim=-1)
        #         pps_center = torch.stack([pps_center_l, pps_center_r, pps_center_t, pps_center_d], dim=2)
        #         pps_wh = pps_wh.unsqueeze(2).expand(pps_center.shape)
        #         pps = torch.cat([pps_center, pps_wh], dim=-1)
        #         pps = pps.reshape(pps.shape[0], -1, 4)
        #         pps = bbox_cxcywh_to_xyxy(pps)
        #         pps_new.append(pps.reshape(*pps_old.shape[0:2], -1, 4))
        #         print(pps.shape)
        #         exit()
        #     pps_new = torch.cat(pps_new, dim=2)
        # else:
        #     pps_new = pps_old
        # h, w, _ = img_meta[i]['img_shape']
        # if cut_mode is 'clamp':
        #     pps_new[..., 0:4:2] = torch.clamp(pps_new[..., 0:4:2], 0, w)
        #     pps_new[..., 1:4:2] = torch.clamp(pps_new[..., 1:4:2], 0, h)
        #     proposals_valid_list.append(pps_new.new_full(
        #         (*pps_new.shape[0:3], 1), 1, dtype=torch.long).reshape(-1, 1))
        # else:
        #     img_xyxy = pps_new.new_tensor([0, 0, w, h])
        #     iof_in_img = bbox_overlaps(pps_new.reshape(-1, 4), img_xyxy.unsqueeze(0), mode='iof')
        #     proposals_valid = iof_in_img > 0.7
        
        # proposals_valid = pps_new.new_full(
        #     (*pps_new.shape[:-1], 1), 1, dtype=torch.long).reshape(-1, 1)
        
        # proposals_valid_list.append(proposals_valid)
        # proposal_list.append(pps_new.reshape(-1, 4))

    return proposal_list, proposals_valid_list


def gen_proposals_from_cfg(gt_points, proposal_cfg, img_meta):
    base_scales = proposal_cfg['base_scales']
    base_ratios = proposal_cfg['base_ratios']
    shake_ratio = proposal_cfg['shake_ratio']
    if 'cut_mode' in proposal_cfg:
        cut_mode = proposal_cfg['cut_mode']
    else:
        cut_mode = 'symmetry'
    base_proposal_list = []
    proposals_valid_list = []
    for i in range(len(gt_points)):
        img_h, img_w, _ = img_meta[i]['img_shape']
        base = min(img_w, img_h) / 100
        base_proposals = []
        for scale in base_scales:
            scale = scale * base
            for ratio in base_ratios:
                base_proposals.append(gt_points[i].new_tensor([[scale * ratio, scale / ratio]]))

        base_proposals = torch.cat(base_proposals)
        base_proposals = base_proposals.repeat((len(gt_points[i]), 1))
        base_center = torch.repeat_interleave(gt_points[i], len(base_scales) * len(base_ratios), dim=0)

        if shake_ratio is not None:
            base_x_l = base_center[:, 0] - shake_ratio * base_proposals[:, 0]
            base_x_r = base_center[:, 0] + shake_ratio * base_proposals[:, 0]
            base_y_t = base_center[:, 1] - shake_ratio * base_proposals[:, 1]
            base_y_d = base_center[:, 1] + shake_ratio * base_proposals[:, 1]
            if cut_mode is not None:
                base_x_l = torch.clamp(base_x_l, 1, img_w - 1)
                base_x_r = torch.clamp(base_x_r, 1, img_w - 1)
                base_y_t = torch.clamp(base_y_t, 1, img_h - 1)
                base_y_d = torch.clamp(base_y_d, 1, img_h - 1)

            base_center_l = torch.stack([base_x_l, base_center[:, 1]], dim=1)
            base_center_r = torch.stack([base_x_r, base_center[:, 1]], dim=1)
            base_center_t = torch.stack([base_center[:, 0], base_y_t], dim=1)
            base_center_d = torch.stack([base_center[:, 0], base_y_d], dim=1)

            shake_mode = 0
            if shake_mode == 0:
                base_proposals = base_proposals.unsqueeze(1).repeat((1, 5, 1))
            elif shake_mode == 1:
                base_proposals_l = torch.stack([((base_center[:, 0] - base_x_l) * 2 + base_proposals[:, 0]),
                                                base_proposals[:, 1]], dim=1)
                base_proposals_r = torch.stack([((base_x_r - base_center[:, 0]) * 2 + base_proposals[:, 0]),
                                                base_proposals[:, 1]], dim=1)
                base_proposals_t = torch.stack([base_proposals[:, 0],
                                                ((base_center[:, 1] - base_y_t) * 2 + base_proposals[:, 1])], dim=1
                                               )
                base_proposals_d = torch.stack([base_proposals[:, 0],
                                                ((base_y_d - base_center[:, 1]) * 2 + base_proposals[:, 1])], dim=1
                                               )
                base_proposals = torch.stack(
                    [base_proposals, base_proposals_l, base_proposals_r, base_proposals_t, base_proposals_d], dim=1)

            base_center = torch.stack([base_center, base_center_l, base_center_r, base_center_t, base_center_d], dim=1)

        if cut_mode == 'symmetry':
            base_proposals[..., 0] = torch.min(base_proposals[..., 0], 2 * base_center[..., 0])
            base_proposals[..., 0] = torch.min(base_proposals[..., 0], 2 * (img_w - base_center[..., 0]))
            base_proposals[..., 1] = torch.min(base_proposals[..., 1], 2 * base_center[..., 1])
            base_proposals[..., 1] = torch.min(base_proposals[..., 1], 2 * (img_h - base_center[..., 1]))

        base_proposals = torch.cat([base_center, base_proposals], dim=-1)
        base_proposals = base_proposals.reshape(-1, 4)
        base_proposals = bbox_cxcywh_to_xyxy(base_proposals)
        proposals_valid = base_proposals.new_full(
            (*base_proposals.shape[:-1], 1), 1, dtype=torch.long).reshape(-1, 1)
        if cut_mode == 'clamp':
            base_proposals[..., 0:4:2] = torch.clamp(base_proposals[..., 0:4:2], 0, img_w)
            base_proposals[..., 1:4:2] = torch.clamp(base_proposals[..., 1:4:2], 0, img_h)
            proposals_valid_list.append(proposals_valid)
        if cut_mode == 'symmetry':
            proposals_valid_list.append(proposals_valid)
        elif cut_mode == 'ignore':
            img_xyxy = base_proposals.new_tensor([0, 0, img_w, img_h])
            iof_in_img = bbox_overlaps(base_proposals, img_xyxy.unsqueeze(0), mode='iof')
            proposals_valid = iof_in_img > 0.7
            proposals_valid_list.append(proposals_valid)
        elif cut_mode is None:
            proposals_valid_list.append(proposals_valid)
        base_proposal_list.append(base_proposals)

    return base_proposal_list, proposals_valid_list


def gen_negative_proposals(gt_points, proposal_cfg, aug_generate_proposals, img_meta):
    num_neg_gen = proposal_cfg['gen_num_neg']
    if num_neg_gen == 0:
        return None, None
    neg_proposal_list = []
    neg_weight_list = []
    for i in range(len(gt_points)):
        pos_box = aug_generate_proposals[i]
        h, w, _ = img_meta[i]['img_shape']
        x1 = -0.2 * w + torch.rand(num_neg_gen) * (1.2 * w)
        y1 = -0.2 * h + torch.rand(num_neg_gen) * (1.2 * h)
        x2 = x1 + torch.rand(num_neg_gen) * (1.2 * w - x1)
        y2 = y1 + torch.rand(num_neg_gen) * (1.2 * h - y1)
        neg_bboxes = torch.stack([x1, y1, x2, y2], dim=1).to(gt_points[0].device)
        gt_point = gt_points[i]
        gt_min_box = torch.cat([gt_point - 10, gt_point + 10], dim=1)
        iou = bbox_overlaps(neg_bboxes, pos_box)
        neg_weight = ((iou < 0.3).sum(dim=1) == iou.shape[1])

        neg_proposal_list.append(neg_bboxes)
        neg_weight_list.append(neg_weight)
    return neg_proposal_list, neg_weight_list



def fine_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg, img_meta, stage):
    gen_mode = fine_proposal_cfg['gen_proposal_mode']
    # cut_mode = fine_proposal_cfg['cut_mode']
    cut_mode = None
    if isinstance(fine_proposal_cfg['base_ratios'], tuple or list):
        base_ratios = fine_proposal_cfg['base_ratios'][stage - 1]
        shake_ratio = fine_proposal_cfg['shake_ratio'][stage - 1]
    else:
        base_ratios = fine_proposal_cfg['base_ratios']
        shake_ratio = fine_proposal_cfg['shake_ratio']
    if gen_mode == 'fix_gen':
        proposal_list = []
        proposals_valid_list = []
        for i in range(len(img_meta)):
            pps = []
            base_boxes = pseudo_boxes[i]
            for ratio_w in base_ratios:
                for ratio_h in base_ratios:
                    base_boxes_ = bbox_xyxy_to_cxcywh(base_boxes)
                    base_boxes_[:, 2] *= ratio_w
                    base_boxes_[:, 3] *= ratio_h
                    base_boxes_ = bbox_cxcywh_to_xyxy(base_boxes_)
                    pps.append(base_boxes_.unsqueeze(1))
            pps_old = torch.cat(pps, dim=1)
            if shake_ratio is not None:
                pps_new = []

                pps_new.append(pps_old.reshape(*pps_old.shape[0:2], -1, 4))
                for ratio in shake_ratio:
                    pps = bbox_xyxy_to_cxcywh(pps_old)
                    pps_center = pps[:, :, :2]
                    pps_wh = pps[:, :, 2:4]
                    pps_x_l = pps_center[:, :, 0] - ratio * pps_wh[:, :, 0]
                    pps_x_r = pps_center[:, :, 0] + ratio * pps_wh[:, :, 0]
                    pps_y_t = pps_center[:, :, 1] - ratio * pps_wh[:, :, 1]
                    pps_y_d = pps_center[:, :, 1] + ratio * pps_wh[:, :, 1]
                    pps_center_l = torch.stack([pps_x_l, pps_center[:, :, 1]], dim=-1)
                    pps_center_r = torch.stack([pps_x_r, pps_center[:, :, 1]], dim=-1)
                    pps_center_t = torch.stack([pps_center[:, :, 0], pps_y_t], dim=-1)
                    pps_center_d = torch.stack([pps_center[:, :, 0], pps_y_d], dim=-1)
                    pps_center = torch.stack([pps_center_l, pps_center_r, pps_center_t, pps_center_d], dim=2)
                    pps_wh = pps_wh.unsqueeze(2).expand(pps_center.shape)
                    pps = torch.cat([pps_center, pps_wh], dim=-1)
                    pps = pps.reshape(pps.shape[0], -1, 4)
                    pps = bbox_cxcywh_to_xyxy(pps)
                    pps_new.append(pps.reshape(*pps_old.shape[0:2], -1, 4))
                pps_new = torch.cat(pps_new, dim=2)
            else:
                pps_new = pps_old
            h, w, _ = img_meta[i]['img_shape']
            if cut_mode is 'clamp':
                pps_new[..., 0:4:2] = torch.clamp(pps_new[..., 0:4:2], 0, w)
                pps_new[..., 1:4:2] = torch.clamp(pps_new[..., 1:4:2], 0, h)
                proposals_valid_list.append(pps_new.new_full(
                    (*pps_new.shape[0:3], 1), 1, dtype=torch.long).reshape(-1, 1))
            else:
                img_xyxy = pps_new.new_tensor([0, 0, w, h])
                iof_in_img = bbox_overlaps(pps_new.reshape(-1, 4), img_xyxy.unsqueeze(0), mode='iof')
                proposals_valid = iof_in_img > 0.7
            proposals_valid_list.append(proposals_valid)
            proposal_list.append(pps_new.reshape(-1, 4))
    return proposal_list, proposals_valid_list



def show_proposals(proposals, img_meta):
    import cv2
    import numpy as np
    filename = img_meta['filename']
    print(filename)
    igs = cv2.imread(filename)
    h, w, _ = img_meta['img_shape']
    igs = cv2.resize(igs, (w, h))
    import copy
    igs1 = copy.deepcopy(igs)
    boxes = np.array(torch.tensor(proposals).cpu()).astype(np.int32)
    for i in range(len(boxes)):
        color = (np.random.randint(0, 255), np.random.randint(0, 255),
                    np.random.randint(0, 255))
        

        for j in range(len(boxes[i])):
            # if neg_weight[i]:

            blk = np.zeros(igs1.shape, np.uint8)
            blk = cv2.rectangle(blk, (boxes[i, j, 0], boxes[i, j, 1]), (boxes[i, j, 2], boxes[i, j, 3]),
                                color=color, thickness=-1)
            # 得到与原图形大小形同的形状

            igs1 = cv2.addWeighted(igs1, 1.0, blk, 0.3, 1, dst=None, dtype=None)
            igs1 = cv2.rectangle(igs1, (boxes[i, j, 0], boxes[i, j, 1]), (boxes[i, j, 2], boxes[i, j, 3]),
                                    color=color, thickness=2)
    save_dir = '/home/ubuntu/Guo/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    cv2.imwrite(save_dir+filename.split('/')[-1],igs1)
            
    
            # cv2.namedWindow("ims1", 0)
            # cv2.resizeWindow("ims1", 2000, 1200)
            # cv2.imshow('ims1', igs1)
            # # cv2.namedWindow("ims", 0)
            # # cv2.resizeWindow("ims", 1333, 800)
            # # cv2.imshow('ims', igs)
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()
            # elif cv2.waitKey(0) & 0xFF == ord('b'):
            #     break

@DETECTORS.register_module()
class SAM_PRNetv5(TwoStageDetector):
    def __init__(self,
                 backbone,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 bbox_head=None,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(SAM_PRNetv5, self).__init__(
            backbone=backbone,
            neck=neck,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.num_stages = roi_head.num_stages
        if bbox_head is not None:
            self.with_bbox_head = True
            self.bbox_head = build_head(bbox_head)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_true_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        # x = tuple([self.extract_feat(img)[-1]])
        
        x =self.extract_feat(img)
        base_proposal_cfg = self.train_cfg.get('base_proposal',
                                               self.test_cfg.rpn)
        fine_proposal_cfg = self.train_cfg.get('fine_proposal',
                                               self.test_cfg.rpn)
        losses = dict()

        # for i in range(len(gt_bboxes)):
        #     if gt_bboxes[i].shape[0] > 100:
        #         gt_bboxes[i] = gt_bboxes[i][:10]

        gt_points = [bbox_xyxy_to_cxcywh(b)[:, :2] for b in gt_bboxes]
        
        # for i in range(len(gt_points)):
        #     if gt_points[i].shape[0] > 100:
        #         gt_points[i] = gt_points[i][:10]

        for stage in range(self.num_stages):
            if stage == 0:
                generate_proposals, proposals_valid_list = gen_proposals_from_seed(gt_bboxes, base_proposal_cfg,
                                                                                  img_meta=img_metas)
                dynamic_weight = torch.cat(gt_labels).new_ones(len(torch.cat(gt_labels)))   # 对损失加权，第一阶段都是1，第二阶段用第一阶段的得分（cls_score*ins_score）。
                
                neg_proposal_list, neg_weight_list = None, None
                pseudo_boxes = generate_proposals
            elif stage == 1:
                
                # proposals_valid_list = []
                # generate_proposals = []
                # for i in range(len(filtered_bboxes)):
                #     pps = filtered_bboxes[i]   # num_gt, 35, 4
                #     proposals_valid = pps.new_full(
                #     (*pps.shape[:-1], 1), 1, dtype=torch.long).reshape(-1, 1)
                #     proposals_valid_list.append(proposals_valid)
                #     generate_proposals.append(pps.reshape(-1, 4))
                
                generate_proposals, proposals_valid_list = fine_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg,
                                                                                    img_meta=img_metas,
                                                                                    stage=stage)
                
                neg_proposal_list, neg_weight_list = gen_negative_proposals(gt_points, fine_proposal_cfg,
                                                                            generate_proposals,
                                                                            img_meta=img_metas)
                
            # else:
            #     generate_proposals, proposals_valid_list = fine_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg,
            #                                                                         img_meta=img_metas,
            #                                                                         stage=stage)
            #     neg_proposal_list, neg_weight_list = gen_negative_proposals(gt_points, fine_proposal_cfg,
            #                                                                 generate_proposals,
            #                                                                 img_meta=img_metas)

            roi_losses, pseudo_boxes, filtered_bboxes, dynamic_weight = self.roi_head.forward_train(stage, x, img_metas,
                                                                                   pseudo_boxes,
                                                                                   generate_proposals,
                                                                                   proposals_valid_list,
                                                                                   neg_proposal_list, neg_weight_list,
                                                                                   gt_true_bboxes, gt_labels,
                                                                                   dynamic_weight,
                                                                                   gt_bboxes_ignore, gt_masks,
                                                                                   **kwargs)
            
            if stage == 0:
                pseudo_boxes_out = pseudo_boxes
                dynamic_weight_out = dynamic_weight
            for key, value in roi_losses.items():
                losses[f'stage{stage}_{key}'] = value
        return losses

    def simple_test(self, img, img_metas, gt_bboxes, gt_anns_id, gt_true_bboxes, gt_labels,
                    gt_bboxes_ignore=None, proposals=None, rescale=False):
        """Test without augmentation."""
        base_proposal_cfg = self.train_cfg.get('base_proposal',
                                               self.test_cfg.rpn)
        fine_proposal_cfg = self.train_cfg.get('fine_proposal',
                                               self.test_cfg.rpn)
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        
        for stage in range(self.num_stages):
            gt_points = [bbox_xyxy_to_cxcywh(b)[:, :2] for b in gt_bboxes]
            if stage == 0:
                generate_proposals, proposals_valid_list = gen_proposals_from_seed(gt_bboxes, base_proposal_cfg,
                                                                                  img_meta=img_metas)
            elif stage == 1:
                # proposals_valid_list = []
                # generate_proposals = []
                # for i in range(len(filtered_bboxes)):
                #     pps = filtered_bboxes[i]   # num_gt, 35, 4
                #     proposals_valid = pps.new_full(
                #     (*pps.shape[:-1], 1), 1, dtype=torch.long).reshape(-1, 1)
                #     proposals_valid_list.append(proposals_valid)
                #     generate_proposals.append(pps.reshape(-1, 4))
                
                generate_proposals, proposals_valid_list = fine_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg,
                                                                                    img_meta=img_metas,
                                                                                    stage=stage)
            else:
                generate_proposals, proposals_valid_list = fine_proposals_from_cfg(pseudo_boxes, fine_proposal_cfg,
                                                                                   img_meta=img_metas, stage=stage)

            test_result, pseudo_boxes, filtered_bboxes = self.roi_head.simple_test(stage,
                                                                  x, generate_proposals, proposals_valid_list,
                                                                  gt_true_bboxes, gt_labels,
                                                                  gt_anns_id,
                                                                  img_metas,
                                                                  rescale=rescale)
            
        return test_result

