from hydra.utils import instantiate
import cfm.clip_dinoiser_backbone.clip_dinoiser  # need to import models here so that they are defined before building
import cfm.clip_dinoiser_backbone.maskclip

def build_model(config, class_names):
    print(f"Building model with config: {config}")
    return instantiate(config, class_names=class_names)