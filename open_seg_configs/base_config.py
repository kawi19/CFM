
model = dict(
    type='CFM_Segmentor',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        bgr_to_rgb=True, 
        # mean=[0., 0., 0.], # model handles internally
        # std=[1., 1., 1.],
        # pad_val=0,           # Pads the image with black pixels
        # seg_pad_val=255,     # Pads the ground truth mask with the ignore index
        # size=(448, 448)      # Forces any image smaller than this to be padded
    ),
    upsampler="anyup",  # "anyup" or "bilinear"
    test_cfg=dict(
        mode='slide',
        stride=(224, 224),
        crop_size=(448, 448)
    )
)

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
    # type='SegLocalVisualizer', vis_backends=vis_backends, alpha=1.0, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=1))



