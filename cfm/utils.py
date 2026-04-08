# import sys
# TODO
import glob
import torchvision
import os.path as osp
import torch
import random
import numpy as np
import cfm.config as config
from pathlib import Path
from cfm.data_utils import probe_classnames
import os

import torch.utils
from torchvision import transforms
from hydra import compose, initialize

from cfm.clip_dinoiser_backbone.builder import build_model
# from cfm.segmentation.datasets import PascalVOCDataset

from cfm.data_utils.cc12m import CC12MImg, CustomDataCollatorImg

initialize(config_path="clip_dinoiser_backbone/configs", version_base=None)

def get_img_model(args, class_names = ["dummy"]):
    if args.img_enc_name.startswith('clip'):
        import clip
        model, preprocess = clip.load(
            args.img_enc_name[5:], device=args.device)
    elif args.img_enc_name.startswith('dinoclip'):
        # preprocess for dinoclip without normalizing since these models do it themselves
        preprocess = None
        if args.img_enc_name == "dinoclip_ViT-B/16":
            preprocess = transforms.Compose([ # preprocess for CLIP ViT-B 16 without normalizing since the model handles that
                            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            # transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275]),
                        ])
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            check_path = os.path.join(BASE_DIR, 'clip_dinoiser_backbone/checkpoints/last.pt')
            check = torch.load(check_path)
            # dinoclip_cfg = os.path(BASE_DIR, '../clip_dinoiser/configs/clip_dinoiser.yaml')
            dinoclip_cfg = "clip_dinoiser.yaml"
            cfg = compose(config_name=dinoclip_cfg)
            device = "cuda"
            print(cfg)
            model = build_model(cfg.model, class_names).to(device)
            # model.clip_backbone.decode_head.use_templates=False # switching off the imagenet templates for fast inference
            model.load_state_dict(check['model_state_dict'], strict=False)
            print(f"Loaded state dict from {check_path}")
            model = model.eval()
        elif args.img_enc_name == "dinoclip_openai_ViT-B/16":
            preprocess = transforms.Compose([ # preprocess for CLIP ViT-B 16 without normalizing
                            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            # transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275]),
                        ])
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            check_path = os.path.join(BASE_DIR, 'clip_dinoiser_backbone/checkpoints_openai/last.pt')
            check = torch.load(check_path, map_location='cpu')
            # dinoclip_cfg = os.path(BASE_DIR, '../clip_dinoiser/configs/clip_dinoiser.yaml')
            dinoclip_cfg = "clip_dinoiser_openai.yaml"
            cfg = compose(config_name=dinoclip_cfg)
            device = "cuda"
            print(cfg)
            model = build_model(cfg.model, class_names).to(device)
            model.clip_backbone.decode_head.use_templates=False # switching off the imagenet templates for fast inference
            model.load_state_dict(check['model_state_dict'], strict=False)
            model = model.eval()
        else:
            raise NotImplementedError
    else: 
        raise NotImplementedError
    return model, preprocess


def get_sae_ckpt(args, autoencoder):
    """
    Loads the SAE checkpoint given configuration in args
    """
    save_dir_ckpt = args.save_dir_sae_ckpts["img"]
    ckpt_path = osp.join(save_dir_ckpt, f'sparse_autoencoder_final.pt')
    print(f"Loading SAE checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path)
    autoencoder.load_state_dict(state_dict)
    return autoencoder


def get_probe_classifier_ckpt(args, which_ckpt=None, name_only=False):
    """
    Loads and returns the probe classifier checkpoint and filename, given args
    """
    if which_ckpt is None:
        which_ckpt = args.probe_classifier_which_ckpt

    checkpoint_save_path = osp.join(
        args.probe_cs_save_dir, args.probe_config_name, "on_concepts_ckpts")

    whole_ckpt_fname = osp.join(
        checkpoint_save_path, f"on_concepts_{which_ckpt}_{args.probe_config_name}.pt")
    if not name_only:
        print(f"Loading classifier checkpoint from: {checkpoint_save_path}")
        state_dict = torch.load(whole_ckpt_fname)

        return state_dict, whole_ckpt_fname
    else:
        return whole_ckpt_fname


def set_seed(seed):
    """
    Set seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_printable_class_name(probe_dataset, class_idx):
    """
    Returns cleaned up class names for visualizations
    """
    if probe_dataset == "places365":
        class_name = " ".join(
            probe_classnames.probe_classes_dict[probe_dataset][class_idx].split("/")[2:])
        class_name = " ".join(class_name.split("_")).capitalize()
    elif probe_dataset == "imagenet":
        class_name = probe_classnames.imagenet_classes_clip[class_idx]
        class_name = class_name.capitalize()
    else:
        class_name = probe_classnames.probe_classes_dict[probe_dataset][class_idx]
        class_name = class_name.capitalize()
    return class_name


def common_init(args, disable_make_dirs=False):
    """
    Performs initializations of variables common to several scripts, and creates directories where applicable
    """

    set_seed(args.seed)
    if args.sae_type == "matryoshka":
        if not(args.no_bias):
            args.config_name = f"k_{args.k}_ef_{args.expansion_factor}_lr_{args.lr}_mf_{args.matryoshka_fractions}"
        else:
            args.config_name = f"k_{args.k}_ef_{args.expansion_factor}_lr_{args.lr}_mf_{args.matryoshka_fractions}_nobias"
    else:
        raise NotImplementedError

    args.img_enc_name_for_saving = args.img_enc_name.replace('/', '')

    # Directory names
    args.autoencoder_input_dim_dict = config.autoencoder_input_dim_dict
    # args.data_dir_root_BS = config.data_dir_root_BS
    args.data_dir_root = config.data_dir_root
    args.save_dir_root = config.save_dir_root
    args.probe_cs_save_dir_root = config.probe_cs_save_dir_root
    args.vocab_dir  = config.vocab_dir
    args.analysis_dir = config.analysis_dir
    args.config_dir = config.config_dir

    args.data_dir_activations = {}
    args.data_dir_activations["img"] = osp.join(
        args.data_dir_root, 'activations_img', args.sae_dataset, args.img_enc_name_for_saving, args.hook_points[0])

    args.probe_data_dir_activations = {}
    args.probe_data_dir_activations["img"] = osp.join(
        args.data_dir_root, 'activations_img', args.probe_dataset, args.img_enc_name_for_saving, args.hook_points[0])

    args.probe_split_idxs_dir = {}
    args.probe_split_idxs_dir["img"] = osp.join(
        args.data_dir_root, 'activations_img', args.probe_dataset)
    
    args.ae_input_dim_dict_key = {}
    args.ae_input_dim_dict_key["img"] = f"{args.img_enc_name_for_saving}"

    args.save_dir = {}
    args.save_dir_sae_ckpts = {}

    # args.save_dir["img"] = Path(osp.join(
    #     args.save_dir_root, f"SAEImg/{args.sae_dataset}/{args.img_enc_name_for_saving}/{args.hook_points[0]}/{args.config_name}"))
    args.save_dir["img"] = Path(osp.join(
         args.save_dir_root, f"{args.sae_dataset}/{args.img_enc_name_for_saving}/{args.sae_type}"))

    if not disable_make_dirs:
        os.makedirs(osp.join(args.save_dir_root,
                    f"{args.sae_dataset}/{args.img_enc_name_for_saving}/{args.sae_type}"), exist_ok=True)
        # os.makedirs(osp.join(args.save_dir_root, f"SAEText/{args.sae_dataset}/{args.text_enc_name_for_saving}/{args.hook_points[0]}"), exist_ok=True)
    modality = "img"
    for modality in args.save_dir:
        if not disable_make_dirs:
            args.save_dir[modality].mkdir(exist_ok=True)
        args.save_dir_sae_ckpts[modality] = Path(
            osp.join(args.save_dir[modality], "sae_checkpoints"))

        if not disable_make_dirs:
            args.save_dir_sae_ckpts[modality].mkdir(exist_ok=True)

    args.enc_name = {}
    args.enc_name["img"] = args.img_enc_name
    args.enc_name_for_saving = {}
    args.enc_name_for_saving["img"] = args.img_enc_name_for_saving

    # bias_str = "nobias"

    if args.probe_classification_loss == "CE" and args.probe_sparsity_loss is None:
        args.probe_config_name = f"lr{args.probe_lr}_bs{args.probe_train_bs}_epo{args.probe_epochs}"
    else:
        args.probe_config_name = f"lr{args.probe_lr}_bs{args.probe_train_bs}_epo{args.probe_epochs}_cl{args.probe_classification_loss}_sp{args.probe_sparsity_loss}_spl{args.probe_sparsity_loss_lambda}"

    args.probe_dataset_root_dir = config.probe_dataset_root_dir_dict[args.probe_dataset]

    args.probe_features_save_dir = osp.join(
        config.probe_cs_save_dir_root, args.sae_dataset, args.img_enc_name_for_saving, args.hook_points[0], "on_features", args.probe_dataset)

    args.probe_cs_save_dir = osp.join(
        config.probe_cs_save_dir_root, args.sae_dataset, args.img_enc_name_for_saving, args.hook_points[0], args.config_name, args.probe_dataset)

    args.probe_labels_dir = {}
    args.probe_labels_dir['img'] = osp.join(
        args.data_dir_root, 'activations_img', args.probe_dataset)

    args.probe_nclasses = config.probe_dataset_nclasses_dict[args.probe_dataset]

    args.probe_config_name_csv = f"{args.probe_lr},{args.probe_train_bs},{args.probe_epochs},{args.probe_classification_loss},{args.probe_sparsity_loss},{args.probe_sparsity_loss_lambda}"
    args.probe_csv_path = osp.join(config.probe_cs_save_dir_root, 'probe_results.csv')

def get_probe_dataset(probe_dataset, probe_split, probe_dataset_root_dir, preprocess_fn, split_idxs=None):
    """
    Loads and returns a downstream dataset given the dataset name, split, root directory, and preprocessing function
    """
    if probe_dataset.startswith("imagenet"):
        if probe_split == "val":
            probe_split = "val_categorized"
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(probe_dataset_root_dir, probe_split), transform=preprocess_fn)
    elif probe_dataset == "places365":
        if probe_split == 'train':
            suffix = '-standard'
        else:
            suffix = ''
        dataset = torchvision.datasets.Places365(
            root=os.path.join(probe_dataset_root_dir), split=f"{probe_split}{suffix}", download=False, small=True, transform=preprocess_fn)
    elif probe_dataset == "cc12m":
        cc12m = CC12MImg()
        collator = CustomDataCollatorImg()
        if probe_split == "train":
            input_shards = sorted(glob.glob(probe_dataset_root_dir + '/cc12m-train-*.tar'))
        else:
            print("cc12m does not have a val split")
            raise NotImplementedError
        dataset = cc12m.get_wds_dataset(input_shards, preprocess_fn, batch_size=1, collator=collator)
    elif probe_dataset == "coco":
        from PIL import Image
        from torch.utils.data import Dataset

        class ImageOnlyDataset(Dataset):
            def __init__(self, root, transform=None):
                self.root = root
                self.transform = transform
                # Filter for images and keep the full paths in a 'samples' list
                valid_extensions = ('.png', '.jpg', '.jpeg')
                self.filenames = sorted([f for f in os.listdir(root) if f.lower().endswith(valid_extensions)])
                # Create a .samples attribute to match ImageFolder behavior: (path, dummy_label)
                self.samples = [(os.path.join(self.root, f), 0) for f in self.filenames]

            def __getitem__(self, index):
                path, _ = self.samples[index]
                img = Image.open(path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img, 0 

            def __len__(self):
                return len(self.samples)

        folder_name = f"{probe_split}2017"
        img_dir = os.path.join(probe_dataset_root_dir, 'images', folder_name)
        dataset = ImageOnlyDataset(root=img_dir, transform=preprocess_fn)
    else:
        raise NotImplementedError
    if split_idxs is not None:
        dataset = torch.utils.data.Subset(dataset, split_idxs)
    return dataset
