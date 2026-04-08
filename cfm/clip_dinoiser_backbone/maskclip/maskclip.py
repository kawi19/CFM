# ---------------------------------------------------------------------------------------------------
# This code is adapted from:
# CLIP-DINOiser
# authors: Monika Wysoczanska, Warsaw University of Technology

# Copyright (c) OpenMMLab. All rights reserved.
# Modified version of the original MaskCLIP code: https://github.com/chongzhou96/MaskCLIP/tree/master
# ---------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from torch import Tensor
from open_clip import get_tokenizer,  create_model_from_pretrained
import torchvision.transforms as T
from .utils.prompt_templates import imagenet_templates
from importlib.metadata import version
OPENAI_NORMALIZE = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


class MaskClip(nn.Module):
    def __init__(
            self,
            backbone,
            decode_head,
            clip_model,
            class_names
        ):
        super(MaskClip, self).__init__()

        self.decode_head = eval(decode_head.get('type'))(clip_model, class_names, **decode_head)
        self.patch_size = backbone.get('patch_size')
        self.img_size = tuple([backbone.get('img_size', 224)]*2)
        pretrained = decode_head.get("pretrained")
        self.is_openai_clip = (pretrained == "openai")
        
        if self.is_openai_clip:
            import clip 
            print("Using OpenAI CLIP model")
            print("CLIP model:", clip_model)
            model, _ = clip.load(clip_model, device='cpu')
            self.is_patch_first = True   # 
        else:
            installed_version = version('open-clip-torch')
            major_version = int(installed_version.split('.')[0])
            self.is_patch_first = major_version<3 # open_clip >= 3.x uses batch-first format for transformer, which changes the shape of the features we hook
            print(f"Using openclip CLIP model weights pretrained on {pretrained}")
            model, _ = create_model_from_pretrained(clip_model, pretrained=pretrained)
        model.eval()
        self.clip_T = OPENAI_NORMALIZE
        self.hook_features = {}
        self.backbone = model
        def hook_fn_forward(module, input, output):
            # self.hook_features["v"] = output.permute(1,0,2) # have to permute for open_clip >= 3.x since there its batch-first 
            if not(self.is_patch_first): 
                # print("Permuting output features for open_clip >= 3.x")
                self.hook_features["v"] = output.permute(1,0,2)
            else:
                self.hook_features["v"] = output
        self.backbone.visual.transformer.resblocks[-2].register_forward_hook(hook_fn_forward)
        self._positional_embd = nn.Parameter(self.backbone.visual.positional_embedding.data.clone())

    # @torch.no_grad() for attribution method
    def extract_feat(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features from images."""
        pos_embed = self.backbone.visual.positional_embedding

        B, C, H, W = inputs.shape
        hw_shape = (H // self.patch_size, W // self.patch_size)
        x_len, pos_len = hw_shape[0]*hw_shape[1], pos_embed.shape[0]

        if x_len != pos_len:
            if pos_len == (self.img_size[0] // self.patch_size) * (self.img_size[1] // self.patch_size) + 1:
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError(
                    '{}, {}'.format(x_len, pos_len))

            self.backbone.visual.positional_embedding.data = self.resize_pos_embed(
                self._positional_embd[None], hw_shape,  (pos_h, pos_w), 'bicubic')[0]

        # Use encode_image() for OpenAI CLIP, forward() for open_clip
        if self.is_openai_clip:
            _ = self.backbone.encode_image(inputs)
        else:
            _ = self.backbone(inputs)
        
        v = self.hook_features["v"]
        v = self.extract_v(v, self.backbone.visual.transformer.resblocks[-1]).permute(1, 0, 2) 
        v = self.backbone.visual.ln_post(v)
        v = v[:, 1:] # needs to be [B, Tokens, feats] remove cls token here
        v = v.reshape(B, hw_shape[0], hw_shape[1], -1).permute(0, 3, 1, 2).contiguous()

        self.backbone.visual.positional_embedding.data = self._positional_embd
        return v

    def extract_v(self, x, block):
        y = block.ln_1(x)
        y = torch.nn.functional.linear(y, block.attn.in_proj_weight, block.attn.in_proj_bias)
        B, N, C = y.shape
        y = y.view(B, N, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * B, N, C // 3)
        y = F.linear(y, block.attn.out_proj.weight, block.attn.out_proj.bias)
        q, k, v = y.tensor_split(3, dim=0)
        v = v + x
        v = v + block.mlp(block.ln_2(v))
        return v


    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = F.interpolate(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    def forward(self, inputs: Tensor, return_feat=False) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        inputs = self.clip_T(inputs)
        x = self.extract_feat(inputs)
        if return_feat:
            seg_logits, feats = self.decode_head(x, return_feat)
            return seg_logits, feats
        else:
            seg_logits = self.decode_head(x)
        return seg_logits
    
    # own implement
    @torch.no_grad()
    def extract_res_cls(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features from images."""
        inputs = self.clip_T(inputs)
        pos_embed = self.backbone.visual.positional_embedding

        B, C, H, W = inputs.shape
        hw_shape = (H // self.patch_size, W // self.patch_size)
        x_len, pos_len = hw_shape[0]*hw_shape[1], pos_embed.shape[0]

        if x_len != pos_len:
            if pos_len == (self.img_size[0] // self.patch_size) * (self.img_size[1] // self.patch_size) + 1:
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError(
                    '{}, {}'.format(x_len, pos_len))

            self.backbone.visual.positional_embedding.data = self.resize_pos_embed(
                self._positional_embd[None], hw_shape,  (pos_h, pos_w), 'bicubic')[0]

        # Use encode_image() for OpenAI CLIP, forward() for open_clip
        if self.is_openai_clip:
            _ = self.backbone.encode_image(inputs)
        else:
            _ = self.backbone(inputs)
        
        v = self.hook_features["v"]
        v = self.extract_v(v, self.backbone.visual.transformer.resblocks[-1]).permute(1, 0, 2)
        v = self.backbone.visual.ln_post(v)
        v = v[:, :1]
        # v = v.unsqueeze(-1).unsqueeze(-1)
        # proj_cls_token = self.decode_head.proj(v)  
        v = v.reshape(B, 1, 1, -1).permute(0, 3, 1, 2).contiguous()
        self.backbone.visual.positional_embedding.data = self._positional_embd
        proj_cls_token = self.decode_head.proj(v)

        return proj_cls_token  # shape: (B, C, 1, 1)

class MaskClipHead(nn.Module):
    def __init__(self, clip_model, class_names, in_channels=3, text_channels=512, use_templates=False, pretrained=None,
                 **kwargs):
        super(MaskClipHead, self).__init__()

        self.text_channels = text_channels
        self.clip_model = clip_model
        self.pretrained = pretrained
        self.class_names = class_names
        self.in_channels = in_channels
        self.use_templates = use_templates

        print(f"Initializing MaskClipHead with CLIP model: {clip_model}, pretrained on: {pretrained}")
        print(f"Using templates: {self.use_templates}")
        
        # Use OpenAI CLIP tokenizer for openai, otherwise use open_clip tokenizer
        if pretrained == "openai":
            import clip
            print("Using OpenAI CLIP model weights pretrained openai in MaskClipHead")
            model, _ = clip.load(clip_model, device='cpu')
            self.tokenizer = clip.tokenize
        else:
            self.tokenizer = get_tokenizer(clip_model)
            model, _ = create_model_from_pretrained(clip_model, pretrained=pretrained)
        
        model.eval()
        self.register_buffer("class_embeddings", self._get_class_embeddings(model, class_names))
        self.proj = nn.Conv2d(self.in_channels, text_channels, 1, bias=False)
        self.proj.weight = nn.Parameter(model.visual.proj.t()[:, :, None, None])

    @torch.no_grad()
    def update_vocab(self, class_names):
        if self.pretrained == "openai":
            import clip 
            print("Using OpenAI CLIP model weights pretrained openai in MaskClipHead update_vocab")
            model, _ = clip.load(self.clip_model, device='cpu')
        else:
            model, _ = create_model_from_pretrained(self.clip_model, pretrained=self.pretrained )
        model.eval()
        self.class_embeddings = self._get_class_embeddings(model, class_names).to(self.proj.weight.device)

    @torch.no_grad()
    def _embed_label(self, text_model: torch.nn.Module, label: str) -> torch.Tensor:
        """
        Encode label name into a single vector
        """
        if self.use_templates:
            templates = imagenet_templates
        elif "laion" in self.pretrained:
            templates = ['a photo of a {}', 'a photo of an {}']
        else:
            templates = ['a {}']
        
        # Handle OpenAI CLIP tokenizer vs open_clip tokenizer
        if self.pretrained == "openai":
            # OpenAI CLIP: tokenizer expects a list of strings
            prompts = [template.format(label) for template in templates]
            all_prompts = self.tokenizer(prompts)
        else:
            # open_clip: tokenizer processes one string at a time
            all_prompts = [self.tokenizer(template.format(label)) for template in templates]
            all_prompts = torch.cat(all_prompts)
        
        out = text_model.encode_text(all_prompts)
        out /= out.norm(dim=-1, keepdim=True)
        out = out.mean(dim=0)
        return out

    def _get_class_embeddings(self, text_model: torch.nn.Module, class_names: List[str]):
        aug_embeddings = torch.stack([self._embed_label(text_model, label) for label in class_names])
        # normalize vector
        aug_embeddings = aug_embeddings / aug_embeddings.norm(dim=-1, keepdim=True)
        return aug_embeddings.squeeze(1)

    def forward(self, inputs, return_feat=False):
        v = inputs
        feat = self.proj(v)
        output = self.cls_seg(feat)
        if return_feat:
            return output, feat
        return output

    def cls_seg(self, feat):
        feat = feat / feat.norm(dim=1, keepdim=True)
        output = F.conv2d(feat, self.class_embeddings[:, :, None, None])
        output = F.softmax(output * 100, dim=1)
        return output
