_base_ = './base.py'
# model settings
model = dict(
    type='WeakRCNN',
    pretrained='torchvision://vgg16',
    backbone=dict(
        type='MyVGG',
        depth=16, 
        out_indices=[4]),
    neck=None,
    roi_head=dict(
        type='OICRRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIPool', output_size=7),
            out_channels=512,
            featmap_strides=[8]),
        bbox_head=dict(
            type='OICRHead',
            in_channels=512,
            hidden_channels=4096,
            roi_feat_size=7,
            num_classes=20))
)

# dataset settings
dataset_type = 'VOCDataset'
data_root = 'data/VOC2012/'
img_norm_cfg = dict(
    mean=[104., 117., 124.], std=[1., 1., 1.], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadWeakAnnotations'),
    dict(type='LoadProposals'),
    dict(type='Resize', img_scale=[(488, 2000), (576, 2000), (688, 2000), (864, 2000), (1200, 2000)], 
         keep_ratio=True, 
         multiscale_mode='value'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_labels', 'proposals']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(688, 2000),
        #img_scale=[(500, 2000), (600, 2000), (700, 2000), (800, 2000), (900, 2000)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'proposals']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/train.txt',
        img_prefix=data_root,
        proposal_file=data_root + 'proposals/SS-voc_2012_train-boxes.pkl',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/val.txt',
        img_prefix=data_root ,
        proposal_file=data_root + 'proposals/SS-voc_2012_val-boxes.pkl',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/Main/val.txt',
        img_prefix=data_root ,
        proposal_file=data_root + 'proposals/SS-voc_2012_val-boxes.pkl',
        pipeline=test_pipeline))

optimizer = dict(
    type='Adam', 
    lr=1e-4,
    weight_decay=0.0005,
    paramwise_cfg=dict(
        bias_decay_mult=0.,
        bias_lr_mult=2.,
        custom_keys={
            'refine': dict(lr_mult=10),
        })
)

work_dir = 'work_dirs/oicr_vgg16/'
