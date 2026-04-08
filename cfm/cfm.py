

import torch
import torch.nn.functional as F
import torchvision.transforms as T

NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

###########################################################################################################################

class CFM(torch.nn.Module):
    """
    CFM - torch.nn.Module with CLIP-DINOiser backbone (feature_extractor) and SAE (autoencoder) for concept mapping.
    """
    def __init__(self, feature_extractor, autoencoder, apply_found = False, device = "cuda"):
        super(CFM, self).__init__()
        self.feature_extractor = feature_extractor.to(device) 
        self.autoencoder = autoencoder.to(device)
        self.apply_found = apply_found     
        self.device = device

    def get_aggregated_concept_activations(self, imgs, construction_method= "max", use_threshold = True):
        "returns concepts in shape (image-level) [batch, concepts]"
        if len(imgs.shape)==3: # in case of unbatched image
            imgs = imgs.unsqueeze(0)
        if construction_method == "max":
            pooled_v = self.feature_extractor.get_pooled_feats(imgs.to(self.device))
            pooled_v_flat = pooled_v.flatten(start_dim=2).permute(0,2,1)
            if not(use_threshold):
                _,_,concept_activations = self.autoencoder.encode(pooled_v_flat, return_active= True, use_threshold = use_threshold)
            else: 
                concept_activations = self.autoencoder.encode(pooled_v_flat)
            concept_activations = concept_activations.permute(0,2,1)
            max_concept_act = concept_activations.max(dim=2)[0]
            return max_concept_act
        else:
            raise NotImplementedError(f"Construction method {construction_method} not implemented")

    def get_concept_activation_map(self, imgs, use_threshold = True):
        "returns concept activation maps (spatial) in shape [batch, concepts, h, w]"
        if len(imgs.shape)==3: # in case of unbatched image input
            imgs = imgs.unsqueeze(0)
        pooled_v = self.feature_extractor.get_pooled_feats(imgs.to(self.device))
        pooled_v_flat = pooled_v.flatten(start_dim=2).permute(0,2,1)
        if not(use_threshold):
            _,_,concept_maps = self.autoencoder.encode(pooled_v_flat, return_active= True, use_threshold = use_threshold)
        else: 
            concept_maps = self.autoencoder.encode(pooled_v_flat)
        concept_maps = concept_maps.permute(0, 2, 1).reshape(concept_maps.shape[0], concept_maps.shape[2],pooled_v.shape[2], pooled_v.shape[3])
        return concept_maps  

    def get_reconstructed_clip_maps(self, imgs, use_threshold = False, use_batch_topk = False, upsampler = None, q_chunk_size = 4096):
        "returns reconstructed clip maps in shape [batch, clip_dim, h, w] and concept activation maps in shape [batch, concepts, h, w]"
        if len(imgs.shape) == 3:  # in case of unbatched image
            imgs = imgs.unsqueeze(0)
        pooled_v = self.feature_extractor.get_pooled_feats(imgs)
        B, C_feat, H, W = pooled_v.shape
        pooled_v_flat = pooled_v.flatten(start_dim=2).permute(0,2,1)
        if use_batch_topk:
            concept_maps,_,_ = self.autoencoder.encode(pooled_v_flat, return_active= True, use_threshold = False)
        else:
            if use_threshold == False:
                _,_,concept_maps = self.autoencoder.encode(pooled_v_flat, return_active= True, use_threshold = use_threshold)
            else: 
                concept_maps = self.autoencoder.encode(pooled_v_flat)

        # 2. Reshape concepts to 2D grid: [B, 8192, 28, 28]
        num_concepts = concept_maps.shape[-1]
        concept_maps_2d = concept_maps.permute(0, 2, 1).reshape(B, num_concepts, H, W)
        # 3. UPSAMPLE THE CONCEPTS
        if upsampler is not None:
            # print("Upsampling CONCEPTS with AnyUp from shape", concept_maps_2d.shape, "to match input image shape", imgs.shape)
            hr_image = NORMALIZE(imgs)  # Normalize the image for AnyUp
            concept_maps_2d = upsampler(hr_image, concept_maps_2d, q_chunk_size=q_chunk_size)
            
            # Update H and W to the new high-res sizes (e.g., 448x448)
            _, _, H, W = concept_maps_2d.shape
        # 4. Flatten the high-res concepts for the decoder
        concept_maps_flat = concept_maps_2d.flatten(start_dim=2).permute(0, 2, 1)
        
        # 5. Decode into high-res CLIP features
        clip_embeds = self.autoencoder.decode(concept_maps_flat)
        
        # 6. Final Reshape
        clip_embeds = clip_embeds.permute(0, 2, 1).reshape(B, -1, H, W)
        
        # print("Reconstructed clip embeds shape:", clip_embeds.shape)
        # print("Concept maps shape:", concept_maps_2d.shape)
        return clip_embeds, concept_maps_2d

    def open_vocab_segment_with_sae_fast(self, x: torch.Tensor, use_threshold = True, batch_topk = False, upsampler = None, q_chunk_size = 4096):
        "returns segmentation output in shape [batch, num_classes, h, w]"
        out_feats, concept_maps = self.get_reconstructed_clip_maps(x, use_threshold, batch_topk, upsampler, q_chunk_size=q_chunk_size)
        # Get the predictions --------------------------------------------------
        output = self.feature_extractor.clip_backbone.decode_head.cls_seg(out_feats)

        if self.apply_found: # background removal
            preds, _, _ = self.feature_extractor.forward_pass(x)
            B, C, hf, wf = output.shape
            if upsampler is not None:
                hr_image = NORMALIZE(x)
                preds = upsampler(hr_image, preds)
            else:
                preds = F.interpolate(preds, (hf, wf), mode="bilinear", align_corners=False)
            # Compute FOUND --------------------------------------------------
            soft_found = torch.sigmoid(preds.detach())
            r_soft_found = soft_found.reshape(-1)
            nb_cls = output.shape[1]
            r_hard_found = (r_soft_found > 0.5).float()
            uncertain = (output.max(dim=1)[0] < self.feature_extractor.delta).reshape(-1)
            output.reshape(1, nb_cls, -1)[:, 0, uncertain & (~r_hard_found.bool())] = 1.0  # background class
        return output
    
    def open_vocab_segment_with_sae_with_contributions(self, x: torch.Tensor, use_threshold = True, upsampler = None, q_chunk_size = 4096):
        "returns segmentation output in shape [batch, num_classes, h, w], and per-concept, per-label spatial contributions in shape [batch, num_concepts, num_classes, h, w]"
        out_feats, concept_maps = self.get_reconstructed_clip_maps(x, use_threshold, upsampler=upsampler, q_chunk_size=q_chunk_size)
        concept_maps_spatial = concept_maps
        concept_maps = concept_maps.flatten(2).permute(0, 2, 1)

        # Compute contribution of each concept to each label (before normalization)
        # W_dec: [8192, 512], class_embeddings: [4, 512]
        concept_to_label = self.autoencoder.W_dec @ self.feature_extractor.clip_backbone.decode_head.class_embeddings.T  # [8192, 4]

        # Expand dimensions to enable broadcasting
        concept_maps_expanded = concept_maps_spatial.unsqueeze(2)  # [1, 8192, 1, 14, 14]
        concept_to_label_expanded = concept_to_label.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, 8192, 4, 1, 1]

        # Multiply to get per-concept, per-label spatial contributions
        concept_label_spatial_contribution = concept_maps_expanded * concept_to_label_expanded
        # concept_to_label_contribution = concept_label_spatial_contribution.sum(dim=(3,4))  # [1, 4, 14, 14] How to get the image wise contribution

        # Get contributions from concepts (before normalization and bias)
        contributions_prenorm = concept_label_spatial_contribution.sum(dim=1)  # [1, 4, 14, 14]

        # Add the bias term contribution
        # b_dec @ class_embeddings.T gives the bias contribution to each label
        bias_contribution = self.autoencoder.b_dec @ self.feature_extractor.clip_backbone.decode_head.class_embeddings.T  # [4]
        contributions_with_bias = contributions_prenorm + bias_contribution[None, :, None, None]  # [1, 4, 14, 14]

        # Now we need to account for the normalization of clip_embeds
        # The normalization scales each spatial position differently
        clip_embeds_unnormalized = concept_maps @ self.autoencoder.W_dec + self.autoencoder.b_dec
        clip_embeds_unnormalized = clip_embeds_unnormalized.permute(0, 2, 1).reshape(
            clip_embeds_unnormalized.shape[0], clip_embeds_unnormalized.shape[2], concept_maps_spatial.shape[2], concept_maps_spatial.shape[3]
        )
        norms = clip_embeds_unnormalized.norm(dim=1, keepdim=True)  # [1, 1, 14, 14]

        # Divide by the norm to match the normalized pipeline
        reconstructed_output = contributions_with_bias / norms  # [1, 4, 14, 14]

        # Apply softmax with temperature (same as original)
        output = F.softmax(reconstructed_output * 100, dim=1)

        if self.apply_found: # background removal
            preds, _, _ = self.feature_extractor.forward_pass(x)
            B, C, hf, wf = output.shape
            if upsampler is not None:
                hr_image = NORMALIZE(x)
                preds = upsampler(hr_image, preds)
            else:
                preds = F.interpolate(preds, (hf, wf), mode="bilinear", align_corners=False)
            # Compute FOUND --------------------------------------------------
            soft_found = torch.sigmoid(preds.detach())
            r_soft_found = soft_found.reshape(-1)
            nb_cls = output.shape[1]
            r_hard_found = (r_soft_found > 0.5).float()
            uncertain = (output.max(dim=1)[0] < self.feature_extractor.delta).reshape(-1)
            output.reshape(1, nb_cls, -1)[:, 0, uncertain & (~r_hard_found.bool())] = 1.0  # background class
        return output, concept_label_spatial_contribution
    
    def memory_efficient_open_vocab_segment_with_sae_with_contributions(self, x: torch.Tensor, use_threshold=True, upsampler=None, q_chunk_size=4096):
        "returns segmentation output in shape [batch, num_classes, h, w], and memory-optimized contributions"
        out_feats, concept_maps = self.get_reconstructed_clip_maps(x, use_threshold, upsampler=upsampler, q_chunk_size=q_chunk_size)
        concept_maps_spatial = concept_maps
        
        B, num_concepts, H, W = concept_maps_spatial.shape

        # W_dec: [8192, 512], class_embeddings: [Num_Classes, 512]
        # concept_to_label: [8192, Num_Classes]
        concept_to_label = self.autoencoder.W_dec @ self.feature_extractor.clip_backbone.decode_head.class_embeddings.T  

        # =========================================================================
        # VRAM OPTIMIZATION 1: Compute logits via Matrix Multiplication
        # Completely skips creating the massive multi-gigabyte intermediate tensor
        # =========================================================================
        num_classes = concept_to_label.shape[1]
        
        # Reshape to [B, H*W, 8192] and multiply by [8192, Num_Classes]
        cm_flat = concept_maps_spatial.view(B, num_concepts, H * W).permute(0, 2, 1)
        
        # Result is [B, Num_Classes, H, W] directly!
        contributions_prenorm = (cm_flat @ concept_to_label).permute(0, 2, 1).view(B, num_classes, H, W)

        # Add the bias term contribution
        bias_contribution = self.autoencoder.b_dec @ self.feature_extractor.clip_backbone.decode_head.class_embeddings.T  
        contributions_with_bias = contributions_prenorm + bias_contribution[None, :, None, None]  

        # Account for the normalization of clip_embeds
        clip_embeds_unnormalized = concept_maps_spatial.flatten(2).permute(0, 2, 1) @ self.autoencoder.W_dec + self.autoencoder.b_dec
        clip_embeds_unnormalized = clip_embeds_unnormalized.permute(0, 2, 1).reshape(B, -1, H, W)
        norms = clip_embeds_unnormalized.norm(dim=1, keepdim=True)  

        # Divide by the norm to match the normalized pipeline
        reconstructed_output = contributions_with_bias / norms  

        # Apply softmax with temperature
        output = F.softmax(reconstructed_output * 100, dim=1)

        # Background Removal (FOUND)
        if self.apply_found: 
            preds, _, _ = self.feature_extractor.forward_pass(x)
            nb_cls = output.shape[1]
            if upsampler is not None:
                # Note: assuming NORMALIZE is imported/available in your scope
                from torchvision.transforms import Normalize
                NORMALIZE = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                hr_image = NORMALIZE(x)
                preds = upsampler(hr_image, preds)
            else:
                preds = F.interpolate(preds, (output.shape[2], output.shape[3]), mode="bilinear", align_corners=False)
                
            soft_found = torch.sigmoid(preds.detach())
            r_soft_found = soft_found.reshape(-1)
            r_hard_found = (r_soft_found > 0.5).float()
            uncertain = (output.max(dim=1)[0] < self.feature_extractor.delta).reshape(-1)
            output.reshape(1, nb_cls, -1)[:, 0, uncertain & (~r_hard_found.bool())] = 1.0  

        # =========================================================================
        # VRAM OPTIMIZATION 2: Pool spatial dimensions BEFORE label mapping
        # Outputs shape [batch, num_concepts, num_classes] instead of a spatial grid
        # =========================================================================
        one_hot_mask = F.one_hot(output.argmax(dim=1).view(B, -1), num_classes=num_classes).float()
        pooled_concepts = torch.bmm(concept_maps_spatial.view(B, num_concepts, -1), one_hot_mask)
        segment_wise_contribution = pooled_concepts * concept_to_label.unsqueeze(0)

        return output, segment_wise_contribution

    def sliding_window_segment_with_contributions(
        self,
        x: torch.Tensor,
        crop_size: int = 448,
        stride: int = 224,
        use_threshold: bool = True,
        upsampler=None,
        q_chunk_size: int = 4096,
    ):
        """
        Sliding-window inference over arbitrarily-sized images using 448×448 crops.

        Logits are averaged over all crops that cover each pixel (standard
        sliding-window segmentation).  Contributions are accumulated as a sum:
        for each crop the raw segment_wise_contribution tensor (shape
        [B, num_concepts, num_classes]) already encodes how many pixels of each
        class that crop contained, so summing across crops naturally weights each
        class by its total pixel coverage across the full image.

        Args:
            x             : Float tensor [B, 3, H, W] or [3, H, W]
            crop_size     : Crop size (pixels) — should equal CLIP's training res (448)
            stride        : Stride between crops; use stride < crop_size for overlap
            use_threshold : Passed to get_reconstructed_clip_maps
            upsampler     : AnyUp module (applied per square crop, so memory is bounded)
            q_chunk_size  : AnyUp chunked-attention chunk size

        Returns:
            seg_logits    : [B, num_classes, H, W] averaged segmentation logits
            contributions : [B, num_concepts, num_classes] accumulated contributions
        """
        import gc
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        B, C, H, W = x.shape

        h_crop = w_crop = crop_size
        h_stride = w_stride = stride

        h_grids = max(H - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(W - w_crop + w_stride - 1, 0) // w_stride + 1

        logits_acc = None        # initialised on first crop
        contrib_acc = None       # [B, num_concepts, num_classes]
        count_mat = x.new_zeros((B, 1, H, W))

        total_crops = h_grids * w_grids
        crop_idx = 0

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                crop_idx += 1
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, H)
                x2 = min(x1 + w_crop, W)
                # clamp so the crop is always exactly crop_size × crop_size
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop = x[:, :, y1:y2, x1:x2]
                # print(f"  Crop {crop_idx}/{total_crops}: "
                #       f"y=[{y1}:{y2}] x=[{x1}:{x2}] shape={tuple(crop.shape)}")

                crop_logits, crop_contribs = \
                    self.memory_efficient_open_vocab_segment_with_sae_with_contributions(
                        crop,
                        use_threshold=use_threshold,
                        upsampler=upsampler,
                        q_chunk_size=q_chunk_size,
                    )
                # crop_logits  : [B, num_classes, h_crop, w_crop]
                # crop_contribs: [B, num_concepts, num_classes]

                if logits_acc is None:
                    num_classes = crop_logits.shape[1]
                    logits_acc = x.new_zeros((B, num_classes, H, W))

                # Pad crop back to full-image canvas and accumulate
                logits_acc += F.pad(
                    crop_logits,
                    (int(x1), int(W - x2), int(y1), int(H - y2)),
                )
                count_mat[:, :, y1:y2, x1:x2] += 1

                # Accumulate contributions — sum is correct because each crop's
                # contribution tensor is already weighted by per-class pixel count
                if contrib_acc is None:
                    contrib_acc = crop_contribs.clone()
                else:
                    contrib_acc = contrib_acc + crop_contribs

                # Free intermediates between crops to avoid VRAM accumulation
                del crop_logits, crop_contribs
                gc.collect()
                torch.cuda.empty_cache()

        assert (count_mat == 0).sum() == 0, \
            "Some pixels were never covered — check crop_size / stride / image size"
        seg_logits = logits_acc / count_mat
        return seg_logits, contrib_acc


