_base_ = './base_config.py'

# model settings
model = dict(
    name_path='../open_seg_configs/cls_voc21.txt',
    slide_crop=384,
    rp_thres=0.15, 
    ap_thres=0.8, 
    with_bkg=True,
    dataset='voc',
)

# dataset settings
dataset_type = 'PascalVOCDataset'
data_root = 'VOC2012' # set this to the path where you downloaded the Pascal VOC 21 dataset

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 448), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline))