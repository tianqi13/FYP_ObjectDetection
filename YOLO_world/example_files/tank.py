from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class TankDataset(CocoDataset):

    METAINFO = {
       'classes': 
       ('object', 'bottle', 'cone', 'cup', 'rubiks cube', 'soda can', 'star', 'valve', 'weight', 'wooden cube') ,

        'palette': [(20, 20, 60), (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70)]
    }