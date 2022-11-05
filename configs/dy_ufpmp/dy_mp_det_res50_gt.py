_base_ = [
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='DyMPDetGT',
    pretrained='torchvision://resnet50',
    scale_path='/home/ranqinglin/work_dirs/ddet/scale_map_net/checkpoints_aug/epoch_14.pth',
    quality_path='/home/ranqinglin/work_dirs/ddet/quality_map_net/checkpoints_aug/epoch_19.pth',
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
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='DYMPHead1GT',
        num_words=200,
        beta=0,
        proxies_list =[2, 3, 2, 5, 4, 8, 8, 4, 3, 3],
        gamma=10,
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)))

# dataset config
dataset_type = 'VisDroneDataset'
data_root = '/data0/ranqinglin/ddet/data/VisDrone_Dataset_COCO_Format/'
# training and testing settings
train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.6),
    max_per_img=500)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    
albu_train_transforms1 = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=45,
        interpolation=1,
        p=0.5)
]

albu_train_transforms2 = [
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadQualityMap'),
    dict(type='LoadScaleMap'),
    dict(type='ResizeWithQualityMapAndScaleMap', img_scale=(1333, 1333), keep_ratio=True),
    dict(type='RandomFlipWithQualityMapAndScaleMap', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PadWithQualityMapAndScaleMap', size_divisor=32),
    dict(
        type='AlbuAdditional',
        transforms=albu_train_transforms1,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            # 'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True,
        additional_targets={
            'quality_map': 'image',
            'scale_map': 'image',
        }),
    dict(
        type='Albu',
        transforms=albu_train_transforms2,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            # 'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='DefaultFormatBundleWithQualityMapAndScaleMap'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'quality_map', 'scale_map']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadQualityMapTest'),
    dict(type='LoadScaleMapTest'),
    dict(
        type='MultiScaleFlipAugAndScalemapAndQualitymap',
        img_scale=(1333, 1333),
        flip=True,
        transforms=[
            #dict(type='Resize', keep_ratio=True),
            dict(type='ResizeWithQualityMapAndScaleMap', keep_ratio=True),
            dict(type='RandomFlipWithQualityMapAndScaleMap'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='PadWithQualityMapAndScaleMap', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img','quality_map', 'scale_map']),
        ])
]


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_UFP_UAVtrain.json',
            img_prefix=data_root + 'images/instance_UFP_UAVtrain/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_UFP_UAVval.json',
        img_prefix=data_root + 'images/instances_UFP_UAVval/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_UFP_UAVval.json',
        img_prefix=data_root + 'images/instances_UFP_UAVval/',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=10, norm_type=2))

custom_hooks = [
    dict(type='OptimalTransportHook', priority='HIGHEST',start_emb=2),
    dict(type='DyConvHook', priority='HIGHEST', per_iter=8089)
]