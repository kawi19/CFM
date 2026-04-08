_base_ = './base_config.py'

# model settings
model = dict(
    name_path='../open_seg_configs/cls_coco_object.txt',
    slide_crop=384,
    rp_thres=0.15, 
    ap_thres=0.8, 
    with_bkg=True,
    dataset='cocoobj'
)

# dataset settings
dataset_type = 'COCOObjectDataset'
data_root = 'coco_stuff164k' # set this to the path where you downloaded the COCO Object dataset

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 448), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
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
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='images/val2017', seg_map_path='annotations/val2017'),
        pipeline=test_pipeline))