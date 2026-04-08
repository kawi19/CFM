autoencoder_input_dim_dict = {'clip_ViT-B16_out': 512,  
                              'dinoclip_ViT-B16' : 512,
                              'dinoclip_openai_ViT-B16': 512,
                              }

# paths for testing
data_dir_root = 'data'
save_dir_root = 'SAE'
probe_cs_save_dir_root = 'probe'
vocab_dir = 'vocab'
analysis_dir = 'analysis'

probe_dataset_root_dir_dict = {
    "places365": "", 
    "imagenet": "", 
    "coco_stuff" : "", 
    "coco" : "",
    "cc12m": "", 
    "cityscapes": "", 
    "cc3m": "", 
}

probe_dataset_nclasses_dict = {"places365": 365, 'imagenet': 1000, "coco_stuff": 171,
                                "coco": 80, "cityscapes": 19, "cc3m": 1, "cc12m": 1}

config_dir = "cfm/clip_dinoiser_backbone/configs"