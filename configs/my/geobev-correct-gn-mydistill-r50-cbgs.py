# Copyright (c) Phigent Robotics. All rights reserved.

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.5, 0.25),
    'rot': (-0, 0),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
    'brightness': (0.8, 1.2),
    'contrast': (1, 1),
    'sharpness': (1, 1),
    'color': (1, 1),
}

# Model
grid_config = {
    'x': [-51.2, 51.2, 0.4],
    'y': [-51.2, 51.2, 0.4],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 0.5],
}

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 80

multi_adj_frame_id_cfg = (1, 1+1, 1)
test_teacher=False
# test_teacher=True

model = dict(
    type='GeoBEVMyDistill',
    scale_num=2,
    loss_distill_weight=1,
    test_teacher=test_teacher,
    use_bev_paste=True,
    bda_aug_conf=bda_aug_conf,
    align_after_view_transfromation=False,
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='RCSampleInBoxMyDistillCorrect',
        scale_num=2,
        keep_threshold=0.1,
        grid_config=grid_config,
        test_teacher=test_teacher,
        input_size=data_config['input_size'],
        ins_channels=[512, 512],
        out_channels=numC_Trans,
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96),
        downsamples=[16, 8],
        depth_loss_cfg=dict(alpha=0.25, gamma=2),
        fg_loss_cfg=dict(alpha=0.25, gamma=2),
        loss_depth_weight=[300, 600],
        loss_fg_weight=[33, 67],),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        # with_cp=True,
        norm_cfg=dict(type='GN', num_groups=8, requires_grad=True),
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        norm_cfg=dict(type='GN', num_groups=8, requires_grad=True),
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    pre_process=dict(
        type='CustomResNet',
        # with_cp=True,
        norm_cfg=dict(type='GN', num_groups=8, requires_grad=True),
        numC_input=numC_Trans,
        num_layer=[2,],
        num_channels=[numC_Trans,],
        stride=[1,],
        backbone_output_ids=[0,]),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        norm_cfg=dict(type='GN', num_groups=8, requires_grad=True),
        tasks=[
            dict(num_class=10, class_names=['car', 'truck',
                                            'construction_vehicle',
                                            'bus', 'trailer',
                                            'barrier',
                                            'motorcycle', 'bicycle',
                                            'pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3,
            norm_cfg=dict(type='GN', num_groups=8, requires_grad=True),),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=6.),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=1.5),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=500,

            # Scale-NMS
            nms_type=['rotate'],
            nms_thr=[0.2],
            nms_rescale_factor=[[1.0, 0.7, 0.7, 0.4, 0.55,
                                 1.1, 1.0, 1.0, 1.5, 3.5]]
        )
    )
)

# Data
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='PrepareImageInputsPreSemMultiScale',
        is_train=True,
        data_config=data_config,
        load_point_label=True,
        load_foreground=False,
        load_timestamps=[0,1],
        img_enhancement=True,
        sequential=True),
    dict(
        type='LoadAnnotationsBEVMyDistill',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d',
                                'gt_bboxes_all', 'gt_labels_all', 'gt_fg', 'gt_depth'
                                ])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    dict(
        type='LoadAnnotationsBEVMyDistill',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs', 'gt_bboxes_all'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR')),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'test']:
    data[key].update(share_data_config)
data['train']['dataset'].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[20,])
runner = dict(type='EpochBasedRunner', max_epochs=20)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        # resume='work_dirs/geobev-mydistill-r50-cbgs/epoch_6_ema.pth',
    ),
    # dict(
    #     type='SequentialControlHook1',
    #     temporal_start_epoch=1,
    #     warmup_ratio=1e-2,
    # ),
    # dict(
    #     type="ProfilerHook",
    #     by_epoch=False,
    #     profile_iters=3,
    #     # on_trace_ready=dict(type="tb_trace", dir_name="hook_prof_logs"),
    #     json_trace_path='profiler-aa.json')
]

fp16 = dict(loss_scale='dynamic')
