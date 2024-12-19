weight = '/fs/atipa/data/cmpe258-sp24/fa24_team14/codys_workspace/PointTransformerV3.1/exp/waymo/semsegv3-waymo-depth-exp/model/model_best.pth'
resume = False
evaluate = True
test_only = False
seed = 6897788
save_path = '/fs/atipa/data/cmpe258-sp24/fa24_team14/codys_workspace/PointTransformerV3.1/exp/waymo/semsegv3-waymo-depth-exp'
num_worker = 32
batch_size = 2
batch_size_val = None
batch_size_test = None
epoch = 10
eval_epoch = 10
clip_grad = None
sync_bn = False
enable_amp = True
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
mix_prob = 0.8
param_dicts = [dict(keyword='block', lr=0.0002)]
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='PreciseEvaluator', test_last=False)
]
train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)
model = dict(
    type='DefaultSegmentorV2',
    num_classes=22,
    backbone_out_channels=64,
    backbone=dict(
        type='PT-v3m1',
        in_channels=4,
        order=['z', 'z-trans', 'hilbert', 'hilbert-trans'],
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 6, 3),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(3, 3, 3, 3),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=('nuScenes', 'SemanticKITTI', 'Waymo')),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ])
optimizer = dict(type='AdamW', lr=0.002, weight_decay=0.005)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=100.0)
dataset_type = 'WaymoDataset'
data_root = 'data/waymo'
ignore_index = -1
names = [
    'Car', 'Truck', 'Bus', 'Other Vehicle', 'Motorcyclist', 'Bicyclist',
    'Pedestrian', 'Sign', 'Traffic Light', 'Pole', 'Construction Cone',
    'Bicycle', 'Motorcycle', 'Building', 'Vegetation', 'Tree Trunk', 'Curb',
    'Road', 'Lane Marker', 'Other Ground', 'Walkable', 'Sidewalk'
]
data = dict(
    num_classes=22,
    ignore_index=-1,
    names=[
        'Car', 'Truck', 'Bus', 'Other Vehicle', 'Motorcyclist', 'Bicyclist',
        'Pedestrian', 'Sign', 'Traffic Light', 'Pole', 'Construction Cone',
        'Bicycle', 'Motorcycle', 'Building', 'Vegetation', 'Tree Trunk',
        'Curb', 'Road', 'Lane Marker', 'Other Ground', 'Walkable', 'Sidewalk'
    ],
    train=dict(
        type='WaymoDataset',
        split='training',
        data_root='data/waymo',
        transform=[
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(
                type='PointClip',
                point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'strength', 'segment'),
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=('coord', 'strength'))
        ],
        test_mode=False,
        ignore_index=-1,
        loop=1),
    val=dict(
        type='WaymoDataset',
        split='validation',
        data_root='data/waymo',
        transform=[
            dict(
                type='PointClip',
                point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
            dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'strength', 'segment'),
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=('coord', 'strength'))
        ],
        test_mode=False,
        ignore_index=-1),
    test=dict(
        type='WaymoDataset',
        split='validation',
        data_root='data/waymo',
        transform=[
            dict(
                type='PointClip',
                point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
            dict(type='Copy', keys_dict=dict(segment='origin_segment')),
            dict(
                type='GridSample',
                grid_size=0.025,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'strength', 'segment'),
                return_inverse=True)
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='test',
                return_grid_coord=True,
                keys=('coord', 'strength')),
            crop=None,
            post_transform=[
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord', 'index'),
                    feat_keys=('coord', 'strength'))
            ],
            aug_transform=[[{
                'type': 'RandomRotateTargetAngle',
                'angle': [0],
                'axis': 'z',
                'center': [0, 0, 0],
                'p': 1
            }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [0.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }]]),
        ignore_index=-1))
