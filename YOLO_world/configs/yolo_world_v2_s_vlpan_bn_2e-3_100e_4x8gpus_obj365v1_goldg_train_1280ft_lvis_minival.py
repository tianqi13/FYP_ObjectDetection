_base_ = ('/vol/bitbucket/th1422/YOLO-World/vscode/3d/YOLO_world/third_party/mmyolo/configs/yolov8/'
          'yolov8_s_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'],
                      allow_failed_imports=False)
#custom_imports = dict(imports=['mmdet.datasets.CocoDataset'], allow_failed_imports=False)
# hyper-parameters
num_classes = 5
num_training_classes = 5
max_epochs = 25  # Maximum training epochs
close_mosaic_epochs = 2
save_epoch_intervals = 5
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 2e-4
weight_decay = 0.025
train_batch_size_per_gpu = 4
img_scale = (1280,1280)
num_cpu_loaders = 2

#Dataset configurations
dataset_type = 'YOLOv5TankDataset'
classes = ('objects', 'cone', 'cup', 'soda can', 'water bottle' )
data_root = '/vol/bitbucket/th1422/YOLO-World/vscode/3d/YOLO_world/data'
num_of_classes = len(classes)

# model settings
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name='openai/clip-vit-base-patch32',
            frozen_modules=['all'])),
    neck=dict(type='YOLOWorldPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv')),
    bbox_head=dict(type='YOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                    use_bn_head=True,
                                    embed_dims=text_channels,
                                    num_classes=num_of_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_of_classes)))

# dataset settings
text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]
train_pipeline = [
    *_base_.pre_transform,
    dict(type='MultiModalMosaic',
         img_scale=img_scale,
         pad_val=114.0,
         pre_transform=_base_.pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        max_aspect_ratio=_base_.max_aspect_ratio,
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    *_base_.last_transform[:-1],
    *text_transform,
]
train_pipeline_stage2 = [
    *_base_.pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        max_aspect_ratio=_base_.max_aspect_ratio,
        border_val=(114, 114, 114)),
    *_base_.last_transform[:-1],
    *text_transform
]
# DATASETS FOR TRAINING 
# obj365v1_train_dataset = dict(
#     type='MultiModalDataset',
#     dataset=dict(
#         type='YOLOv5Objects365V1Dataset',
#         data_root='data/objects365v1/',
#         ann_file='annotations/objects365_train.json',
#         data_prefix=dict(img='train/'),
#         filter_cfg=dict(filter_empty_gt=False, min_size=32)),
#     class_text_path='data/texts/obj365v1_class_texts.json',
#     pipeline=train_pipeline)

# mg_train_dataset = dict(type='YOLOv5MixedGroundingDataset',
#                         data_root='data/mixed_grounding/',
#                         ann_file='annotations/final_mixed_train_no_coco.json',
#                         data_prefix=dict(img='gqa/images/'),
#                         filter_cfg=dict(filter_empty_gt=False, min_size=32),
#                         pipeline=train_pipeline)

# flickr_train_dataset = dict(
#     type='YOLOv5MixedGroundingDataset',
#     data_root='data/flickr/',
#     ann_file='annotations/final_flickr_separateGT_train.json',
#     data_prefix=dict(img='full_images/'),
#     filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     pipeline=train_pipeline)

# cou_train_dataset = dict(
#     _delete_=True,
#     type='MultiModalDataset',
#     dataset=dict(type=dataset_type,
#                  data_root=data_root,
#                  ann_file='/vol/bitbucket/th1422/YOLO-World/vscode/worldtq/coco/coco/train_annotations.json',
#                  data_prefix=dict(img=''),
#                  filter_cfg=dict(filter_empty_gt=False, min_size=32)
#                 ),
#     class_text_path='/vol/bitbucket/th1422/YOLO-World/vscode/worldtq/coco/coco/class_texts.json',
#     pipeline=train_pipeline)

tank_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(type=dataset_type,
                 data_root=data_root+'/train',
                 ann_file='/vol/bitbucket/th1422/YOLO-World/vscode/3d/YOLO_world/data/train/_annotations.coco.json',
                 data_prefix=dict(img=''),
                 filter_cfg=dict(filter_empty_gt=False, min_size=32)
                ),
    class_text_path='/vol/bitbucket/th1422/YOLO-World/vscode/3d/YOLO_world/data/class_texts.json',
    pipeline=train_pipeline)

train_dataloader = dict(batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=dict(tank_train_dataset))


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))
]

# coco_val_dataset = dict(
#     _delete_=True,
#     type='MultiModalDataset',
#     dataset=dict(type='YOLOv5CocoDataset',
#                  data_root='data/coco/',
#                  test_mode=True,
#                  ann_file='lvis/lvis_v1_minival_inserted_image_name.json',
#                  data_prefix=dict(img=''),
#                  batch_shapes_cfg=None),
#     class_text_path='data/texts/lvis_v1_class_texts.json',
#     pipeline=test_pipeline)

#DATALOADERS
# cou_val_dataset = dict(
#     _delete_=True,
#     type='MultiModalDataset',
#     dataset=dict(type=dataset_type,
#                  data_root=data_root,
#                  test_mode=True,
#                  ann_file='/vol/bitbucket/th1422/YOLO-World/vscode/worldtq/coco/coco/val_annotations.json',
#                  data_prefix=dict(img=''),
#                  ),
#     class_text_path='/vol/bitbucket/th1422/YOLO-World/vscode/worldtq/coco/coco/class_texts.json',
#     pipeline=test_pipeline)

tank_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(type=dataset_type,
                 data_root=data_root+'/valid',
                 test_mode=True,
                 ann_file='/vol/bitbucket/th1422/YOLO-World/vscode/3d/YOLO_world/data/valid/_annotations.coco.json',
                 data_prefix=dict(img=''),
                 ),
    class_text_path='/vol/bitbucket/th1422/YOLO-World/vscode/3d/YOLO_world/data/class_texts.json',
    pipeline=test_pipeline)

val_dataloader = dict(dataset=tank_val_dataset)

# cou_test_dataset = dict(
#     _delete_=True,
#     type='MultiModalDataset',
#     dataset=dict(type=dataset_type,
#                  data_root=data_root,
#                  test_mode=True,
#                  ann_file='/vol/bitbucket/th1422/YOLO-World/vscode/worldtq/coco/coco/test_annotations.json',
#                  data_prefix=dict(img=''),
#                  ),
#     class_text_path='/vol/bitbucket/th1422/YOLO-World/vscode/worldtq/coco/coco/class_texts.json',
#     pipeline=test_pipeline)

tank_test_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(type=dataset_type,
                 data_root=data_root+'/test',
                 test_mode=True,
                 ann_file='/vol/bitbucket/th1422/YOLO-World/vscode/3d/YOLO_world/data/test/_annotations.coco.json',
                 data_prefix=dict(img=''),
                 ),
    class_text_path='/vol/bitbucket/th1422/YOLO-World/vscode/3d/YOLO_world/data/class_texts.json',
    pipeline=test_pipeline)

test_dataloader = dict(dataset=tank_test_dataset)

# EVALUATORS
val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file='/vol/bitbucket/th1422/YOLO-World/vscode/3d/YOLO_world/data/valid/_annotations.coco.json',
    metric='bbox'
)

test_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file='/vol/bitbucket/th1422/YOLO-World/vscode/3d/YOLO_world/data/test/_annotations.coco.json',
    metric='bbox'
)

# training settings
default_hooks = dict(param_scheduler=dict(max_epochs=max_epochs),
                     checkpoint=dict(interval=save_epoch_intervals,
                                     rule='greater'))
custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=train_pipeline_stage2)
]
train_cfg = dict(max_epochs=max_epochs,
                 val_interval=save_epoch_intervals,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])
optim_wrapper = dict(optimizer=dict(
    _delete_=True,
    type='AdamW',
    lr=base_lr,
    weight_decay=weight_decay,
    batch_size_per_gpu=train_batch_size_per_gpu),
                     paramwise_cfg=dict(bias_decay_mult=0.0,
                                        norm_decay_mult=0.0,
                                        custom_keys={
                                            'backbone.text_model':
                                            dict(lr_mult=0.01),
                                            'logit_scale':
                                            dict(weight_decay=0.0)
                                        }),
                     constructor='YOLOWv5OptimizerConstructor')
