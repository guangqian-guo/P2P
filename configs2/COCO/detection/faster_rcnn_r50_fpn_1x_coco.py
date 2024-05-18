_base_ = [
    '../../../configs/_base_/models/faster_rcnn_r50_fpn.py',
    '../../../configs/_base_/schedules/schedule_1x.py', '../../../configs/_base_/default_runtime.py'
]

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)  # batch_size=8 --> lr=0.01
# model settings
model = dict(
    type='FasterRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

dataset_type = 'CocoFmtDataset'
data_root = 'data/COCO/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/pred_annotations/result_with_iou.json',  #SAM pred pse-bbox as GT
        # ann_file='/home/ps/Guo/segment-anything-main/proposals_gen_by_predbox/result_with_iou.json', # P2B+SAM pred pse-bbox as GT
        # ann_file='/home/ps/Guo/segment-anything-main/proposals_gen_by_SAM_PRNet_pred_bbox/results_with_iou.json', # SAM SAM_PRNet SAM pred pse-bbox as GT
        #ann_file='/home/ps/Guo/P2BNet-main/TOV_mmdetection/work-dir/COCO/SAM_PRNetv3_lr0.02/coco_1200_latest_pseudo_ann_all.json', # SAMPRNetv3 pred pse-box as GT
        #ann_file='/home/ps/Guo/P2BNet-main/TOV_mmdetection/work-dir/COCO/SAM_PRNetv3_lr0.02/proposals_by_sam_pointandbox/results_with_iou.json',  # SAMPRNetv3 + SAM refine box as GT
        #ann_file='/home/ps/Guo/P2BNet-main/TOV_mmdetection/work-dir/COCO/SAM_PRNetv3_lr0.02v2/coco_1200_latest_result_all_1st_stage.json',   #
        # ann_file='/home/ps/Guo/P2BNet-main/TOV_mmdetection/work-dir/COCO/SAM_PRNetv5_Headv6_lr0.02/SAM_refine_proposal/results_with_iou_filter.json',
        # ann_file='/home/ps/Guo/Project/P2BNet-main/TOV_mmdetection/work-dir/COCO/SAM_PRNetv7_Headv11_MSEv7_lr0.02/coco_latest_pse_ann_all.json',
        # ann_file='/home/ps/Guo/Project/P2BNet-main/TOV_mmdetection/work-dir/COCO/SAM_PRNetv7_Headv11_MSEv7_lr0.02_thr_0.7_3iter_1.5x/3iter/SAM_predict_proposals/results_with_iou_filter.json',
        # ann_file='/home/ps/Guo/Project/P2BNet-9.19/P2BNet-main/TOV_mmdetection/work-dir/COCO/ablation/p2b_group/coco_latest_pse_ann.json',
        ann_file='work-dir/COCO/ablation/p2b_group/SAM_predicted_results/coco_latest_results_all.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017',
        pipeline=test_pipeline))

# evaluation = dict(interval=3, metric='bbox',do_final_eval=True)
evaluation = dict(interval=3, metric='bbox')
