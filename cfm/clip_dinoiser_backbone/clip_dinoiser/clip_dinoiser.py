#  ---------------------------------------------------------------------------------------------------
# This code is adapted from:
# CLIP-DINOiser
# authors: Monika Wysoczanska, Warsaw University of Technology & Oriane Simeoni, valeo.ai
# ---------------------------------------------------------------------------------------------------

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from omegaconf import OmegaConf
import importlib.metadata
from ..builder import build_model
NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

###########################################################################################################################
class DinoCLIP(nn.Module):
    """
    Base model for all the backbones. Implements CLIP features refinement based on DINO dense features and background
    refinement.

    """

    def __init__(self, clip_backbone, class_names, vit_arch="vit_base", vit_patch_size=16, enc_type_feats="k",
                 gamma=0.2, delta=0.99, apply_found=False):
        super(DinoCLIP, self).__init__()
        self.vit_arch = vit_arch
        self.enc_type_feats = enc_type_feats
        self.gamma = gamma
        self.vit_patch_size = vit_patch_size
        self.apply_found = apply_found
        self.delta = delta
        print(f"Initializing DinoCLIP with apply_found={self.apply_found}, delta={self.delta}, gamma={self.gamma}, vit_arch={self.vit_arch}, vit_patch_size={self.vit_patch_size}, enc_type_feats={self.enc_type_feats}")

        # ==== build MaskCLIP backbone equivalent =====
        # Load config relative to this package so it works from notebooks
        # regardless of the current working directory.
        config_dir = Path(__file__).resolve().parent.parent / "configs"
        maskclip_cfg = OmegaConf.load(str(config_dir / f"{clip_backbone}.yaml"))
        self.clip_backbone = build_model(maskclip_cfg["model"], class_names=class_names)
        for param in self.clip_backbone.parameters():
            param.requires_grad = False

    def make_input_divisible(self, x: torch.Tensor) -> torch.Tensor:
        """Pad some pixels to make the input size divisible by the patch size."""
        B, _, H_0, W_0 = x.shape
        pad_w = (self.vit_patch_size - W_0 % self.vit_patch_size) % self.vit_patch_size
        pad_h = (self.vit_patch_size - H_0 % self.vit_patch_size) % self.vit_patch_size
        x = nn.functional.pad(x, (0, pad_w, 0, pad_h), value=0)
        return x

    @torch.no_grad()
    def extract_feats(self, type_feats="k"):
        """
        DINO feature extractor. Attaches a hook on the last attention layer.
        :param type_feats: (string) - type of features from DINO ViT
        """
        nh = self.vit_encoder.blocks[-1].attn.num_heads
        nb_im, nb_tokens, C_qkv = self.hook_features["qkv"].shape

        qkv = (
            self.hook_features["qkv"]
                .reshape(
                nb_im, nb_tokens, 3, nh, C_qkv // nh // 3
            )  # 3 corresponding to |qkv|
                .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        if type_feats == "q":
            return q.transpose(1, 2).float()
        elif type_feats == "k":
            return k.transpose(1, 2).float()
        elif type_feats == "v":
            return v.transpose(1, 2).float()
        else:
            raise ValueError("Unknown features")

    @torch.no_grad()
    def forward(self, x: torch.Tensor):

        x = self.make_input_divisible(x)
        output, raw_similarities = self.get_clip_features(x)
        B, _, H_feat, W_feat = output.shape
        masks = self.get_dino_corrs(x)
        output = self.compute_weighted_pool(output, masks)
        output = self.clip_backbone.decode_head.cls_seg(output)
        if self.apply_found:
            # Compute FOUND --------------------------------------------------
            preds = self.get_found_preds(x)
            r_soft_found = T.functional.resize(preds, (H_feat, W_feat)).reshape(-1)
            nb_cls = output.shape[1]
            r_hard_found = (r_soft_found > 0.5).float()
            uncertain = (output.max(dim=1)[0] < self.delta).reshape(-1)
            output.reshape(1, nb_cls, -1)[:, 0, uncertain & (~r_hard_found.bool())] = 1.0  # background class
        return output

    @torch.no_grad()
    def get_dino_corrs(self, x: torch.Tensor):
        """
        Gets correlations of DINO features. Applies a threshold on the correlations with self.gamma.

        :param x: (torch.Tensor) - batch of input images
        :return: (torch.Tensor) - feature correlations
        """
        B = x.shape[0]
        feats, (hf, wf) = self.get_dino_features(x)  # B C (h_f * w_f) normalized
        corrs = torch.matmul(feats.permute(0, 2, 1), feats).reshape(B, hf, wf, hf * wf)
        if self.gamma is not None:
            corrs[corrs < self.gamma] = 0.0

        return corrs.permute(0, 3, 1, 2)  # B C h w

    def get_dino_features(self, x: torch.Tensor):
        """
        Extracts dense DINO features.

        :param x: (torch.Tensor) - batch of input images
        :return: (torch.Tensor) - of dense DINO features, (int, int) - size of feature map
        """
        x = self.make_input_divisible(x)
        batch = self.dino_T(x)  # tensor B C H W
        h_featmap = batch.shape[-2] // self.vit_patch_size
        w_featmap = batch.shape[-1] // self.vit_patch_size

        # Forward pass
        # Encoder forward pass and get hooked intermediate values
        _ = self.vit_encoder(batch)

        # Get decoder features
        feats = self.extract_feats(type_feats=self.enc_type_feats)
        num_extra_tokens = 1

        # B nbtokens+1 nh dim
        feats = feats[:, num_extra_tokens:, :, :].flatten(-2, -1).permute(0, 2, 1)  # B C nbtokens
        # B, C, nbtokens
        feats = feats / feats.norm(dim=1, keepdim=True)  # normalize features

        return feats, (h_featmap, w_featmap)

    # @torch.no_grad() removed for gradients for attribution methods
    def get_clip_features(self, x: torch.Tensor):
        """
        Extracts MaskCLIP features
        :param x: (torch.Tensor) - batch of input images
        :return: (torch.Tensor) - clip dense features, (torch.Tensor) - output probabilities
        """
        x = self.make_input_divisible(x)
        maskclip_map, feat = self.clip_backbone(x, return_feat=True)

        return feat, maskclip_map

    @staticmethod
    def compute_weighted_pool(maskclip_feats: torch.Tensor, corrs: torch.Tensor):
        """
        Weighted pooling method.
        :param maskclip_feats: torch.tensor - raw clip features
        :param corrs: torch.tensor - correlations as weights for pooling mechanism
        :return: torch.tensor - refined clip features
        """
        B = maskclip_feats.shape[0]
        h_m, w_m = maskclip_feats.shape[-2:]
        h_w, w_w = corrs.shape[-2:]

        if (h_m != h_w) or (w_m != w_w):
            maskclip_feats = F.interpolate(
                maskclip_feats,
                size=(h_w, w_w),
                mode='bilinear',
                align_corners=False)
            h_m, w_m = h_w, w_w

        maskclip_feats_ref = torch.einsum("bnij, bcij -> bcn", corrs, maskclip_feats)  # B C HW
        norm_factor = corrs.flatten(-2, -1).sum(dim=-1)[:, None]  # B 1 HW
        maskclip_feats_ref = maskclip_feats_ref / (norm_factor + 1e-6)

        # RESHAPE back to 2d
        maskclip_feats_ref = maskclip_feats_ref.reshape(B, -1, h_m, w_m)
        return maskclip_feats_ref

class CLIP_DINOiser(DinoCLIP):
    """
    CLIP-DINOiser backbone- torch.nn.Module with two single conv layers for object correlations (obj_proj) and background
    filtering (bkg_decoder).
    """

    def __init__(self, clip_backbone, class_names, vit_arch="vit_base", vit_patch_size=16, enc_type_feats="v",
                 feats_idx=-3, gamma=0.2, delta=0.99, in_dim=256, conv_kernel=3):
        super(CLIP_DINOiser, self).__init__(clip_backbone, class_names, vit_arch, vit_patch_size, enc_type_feats, gamma)
        if vit_patch_size == 16:
            in_size = 768 if feats_idx != 'final' else 512
        elif vit_patch_size == 14:
            in_size = 1024 if feats_idx != 'final' else 768
        self.gamma = gamma
        self.feats_idx = feats_idx
        self.delta = delta
        self.in_dim = in_dim
        self.bkg_decoder = nn.Conv2d(in_size, 1, (1, 1))
        self.obj_proj = nn.Conv2d(in_size, in_dim, (conv_kernel, conv_kernel), padding=conv_kernel // 2,
                                  padding_mode='replicate')
        self.is_patch_first= self.clip_backbone.is_patch_first

        # setup clip features for training
        if feats_idx != 'final':
            train_feats = {}

            def get_activation(name, is_patch_first= False):
                def hook(model, input, output):
                    # Handle both patch-first (P, B, D) from older OpenCLIP
                    # and batch-first (B, P, D) from newer OpenCLIP versions.
                    # Heuristic: if dim 0 is larger than dim 1, it's likely that its 
                    if is_patch_first:
                        # Likely sequence-first (P, B, D) — older OpenCLIP
                        # print(f"Assuming patch-first format for {name}")
                        train_feats[name] = output.detach().permute(1, 0, 2)
                    else:
                        # Likely already batch-first (B, S, D) — newer OpenCLIP
                        # print(f"Assuming batch-first format for {name}")
                        train_feats[name] = output.detach()
                return hook

            self.clip_backbone.backbone.visual.transformer.resblocks[feats_idx].ln_2.register_forward_hook(
                get_activation('clip_inter', self.is_patch_first))
            self.train_feats = train_feats
    
    def forward_pass(self, x: torch.Tensor):
        x = self.make_input_divisible(x)
        clip_proj_feats = self.get_clip_features(x)[0]
        B, c_dim, h, w = clip_proj_feats.shape
        if self.feats_idx != 'final':
            clip_feats = self.train_feats['clip_inter']
            B, N, c_dim = clip_feats.shape
            clip_feats = clip_feats[:, 1:, ].permute(0, 2, 1).reshape(B, c_dim, h, w)
        else:
            clip_feats = clip_proj_feats
        proj_feats = self.obj_proj(clip_feats).reshape(B, self.in_dim, -1)
        proj_feats = proj_feats / proj_feats.norm(dim=1, keepdim=True)
        corrs = torch.matmul(proj_feats.permute(0, 2, 1), proj_feats).reshape(B, h * w, h, w)
        output = clip_feats / clip_feats.norm(dim=1, keepdim=True)
        bkg_out = self.bkg_decoder(output)

        return bkg_out, corrs, clip_proj_feats

    def forward_pass_without_found(self, x: torch.Tensor):
        x = self.make_input_divisible(x)
        clip_proj_feats = self.get_clip_features(x)[0]
        B, c_dim, h, w = clip_proj_feats.shape
        if self.feats_idx != 'final':
            clip_feats = self.train_feats['clip_inter']
            B, N, c_dim = clip_feats.shape
            clip_feats = clip_feats[:, 1:, ].permute(0, 2, 1).reshape(B, c_dim, h, w)
        else:
            clip_feats = clip_proj_feats
        proj_feats = self.obj_proj(clip_feats).reshape(B, self.in_dim, -1)
        proj_feats = proj_feats / proj_feats.norm(dim=1, keepdim=True)
        corrs = torch.matmul(proj_feats.permute(0, 2, 1), proj_feats).reshape(B, h * w, h, w)
        return  corrs, clip_proj_feats

    def forward(self, x: torch.Tensor, upsampler=None):
        preds, corrs, output = self.forward_pass(x)
        B, C, hf, wf = output.shape
        # Compute weighted pooling --------------------------------------------------
        if self.gamma:
            corrs[corrs < self.gamma] = 0.0
        out_feats = self.compute_weighted_pool(output, corrs)
        # up-sample to original image size if upsampler is provided --------------------------------------------------
        if upsampler is not None:
            print("Upsampling with AnyUp in clip_dinoiser forward from shape", out_feats.shape, "to match input image shape", x.shape)
            hr_image = NORMALIZE(x)  # Normalize the image for AnyUp
            out_feats = upsampler(hr_image, out_feats) 
            

        # Get the predictions --------------------------------------------------
        output = self.clip_backbone.decode_head.cls_seg(out_feats)

        if self.apply_found:
            if upsampler is not None:
                preds = upsampler(hr_image, preds)
            else:
                preds = F.interpolate(preds, (hf, wf), mode="bilinear", align_corners=False)
            # Compute FOUND --------------------------------------------------
            soft_found = torch.sigmoid(preds.detach())
            r_soft_found = soft_found.reshape(-1)
            nb_cls = output.shape[1]
            r_hard_found = (r_soft_found > 0.5).float()
            uncertain = (output.max(dim=1)[0] < self.delta).reshape(-1)
            output.reshape(1, nb_cls, -1)[:, 0, uncertain & (~r_hard_found.bool())] = 1.0  # background class

        return output

    def get_pooled_feats(self, x: torch.Tensor):
        x = self.make_input_divisible(x)
        clip_proj_feats = self.get_clip_features(x)[0]
        B, c_dim, h, w = clip_proj_feats.shape
        if self.feats_idx != 'final':
            clip_feats = self.train_feats['clip_inter']
            B, N, c_dim = clip_feats.shape
            clip_feats = clip_feats[:, 1:, ].permute(0, 2, 1).reshape(B, c_dim, h, w)
        else:
            clip_feats = clip_proj_feats
        proj_feats = self.obj_proj(clip_feats).reshape(B, self.in_dim, -1)
        proj_feats = proj_feats / proj_feats.norm(dim=1, keepdim=True)
        corrs = torch.matmul(proj_feats.permute(0, 2, 1), proj_feats).reshape(B, h * w, h, w)
        output = clip_proj_feats

        # Compute weighted pooling --------------------------------------------------
        if self.gamma:
            corrs[corrs < self.gamma] = 0.0
        out_feats = self.compute_weighted_pool(output, corrs)
        return out_feats

    
        