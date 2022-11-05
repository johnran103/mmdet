_base_ = [
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
mid_ch=256
out_ch=256
# model settings
model = dict(
    type='HRGFL',
    backbone=dict(
        type='Space2DepthChainAdd1',
    ),
    neck=dict(
        type='GiraffeNeck',
        min_level=3,
        max_level=7,
        num_levels=5,
        norm_layer=None,
        norm_kwargs=dict(eps=.001, momentum=.01),
        act_type='silu',
        fpn_config=None,
        fpn_name='giraffeneck',
        weight_method='concat',
        depth_multiplier=7,
        width_multiplier=0.7,
        with_backslash=True,
        with_slash=True,
        with_skip_connect=True,
        skip_connect_type='log2n',
        fpn_channels=[mid_ch, mid_ch, mid_ch, mid_ch, mid_ch],
        out_fpn_channels=[out_ch, out_ch, out_ch, out_ch, out_ch],
        pad_type='',
        downsample_type='max',
        upsample_type='nearest',
        apply_resample_bn=True,
        conv_after_downsample=False,
        redundant_bias=False,
        fpn_cell_repeats=1,
        separable_conv=False,
        conv_bn_relu_pattern=False,
        feature_info=[dict(num_chs=128, reduction=8), dict(num_chs=256, reduction=16), dict(num_chs=512, reduction=32), dict(num_chs=1024, reduction=64), dict(num_chs=2048, reduction=128)],
        alternate_init=False),
    bbox_head=dict(
        type='GFLHead',
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
data_root = '/home/ranqinglin/work_dirs/ddet/data/uav-dataset/VisDrone/'
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
    nms=dict(type='nms', iou_threshold=0.6, class_agnostic=True),
    max_per_img=500)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=45,
        interpolation=1,
        p=0.5),
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
    dict(type='Resize', img_scale=(1300, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=128),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
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
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1300, 800),
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=128),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
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
            ann_file=data_root + 'annotations/instances_UAVtrain.json',
            img_prefix=data_root + 'images/instances_UAVtrain/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_UAVval.json',
        img_prefix=data_root + 'images/instances_UAVval/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_UAVval.json',
        img_prefix=data_root + 'images/instances_UAVval/images/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=10, norm_type=2))
