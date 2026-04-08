"""
Save top activating images per concept to subfolders and compute activation statistics.

This script:
1. Computes activations for all images in the probe dataset
2. Finds top-k activating images for each concept using min-heaps 
3. Saves these images and their activation maps to concept-specific subfolders
4. Computes activation frequency and mean activation magnitude per concept
"""

import os
import os.path as osp
import torch
import torchvision
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time

from cfm.cfm import CFM
from cfm.utils import common_init, get_img_model, get_probe_dataset
from cfm import arg_parser, config
from dictionary_learning.utils import load_dictionary
import imagehash
from PIL import Image
import numpy as np


def get_pil_image_for_hashing(item_data, is_webdataset=True, dataset_obj=None):
    """
    Converts an item from all_activations into a PIL Image for hashing.
    item_data is either a tensor (webdataset) or an index (non-webdataset).
    This is the "hash the img_tensor" step you're asking about.
    """
    if is_webdataset:
        img_tensor = item_data  # item_data is img_tensor
    else:
        img_tensor = dataset_obj[item_data][0]  # item_data is img_idx
    img_tensor = torch.clamp(img_tensor, 0, 1)
    img_np = img_tensor.permute(1, 2, 0).numpy()
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    return img_pil

def compute_activations_and_save_images(
    args, 
    dataset, 
    model, 
    autoencoder, 
    output_dir,
    threshold=0.01, 
    top_k=5,
    construction_method="max",
    hash_threshold = None # For deduplication of top images based on perceptual hash distance (if None, no deduplication applied). 5 is a appropriate value to deduplicate near-duplicates while allowing some variation. Adjust as needed based on dataset and desired strictness.
):
    """
    Two-pass approach for speed:
    Pass 1: Find top-k images per concept using only pooled activations (FAST)
    Pass 2: Compute activation maps only for the top-k images (MINIMAL)
    """
    model.eval()
    autoencoder.eval()

    use_hash_dedup = hash_threshold is not None
    
    
    batch_size = 512 # adjust based on memory constraints
    is_webdataset = args.probe_dataset == "cc12m"
    
    # Setup dataloader
    if is_webdataset:
        from cfm.data_utils import cc12m
        dataloader = cc12m.get_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        print(f"Processing approx. 11,000,000 images in batches of {batch_size}")
        total_images = 11000000
        num_batches = total_images // batch_size
    else:
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        print(f"Processing {len(dataset)} images in batches of {batch_size}")
        total_images = len(dataset)
        num_batches = (total_images + batch_size - 1) // batch_size
    
    # Enable CUDA memory fragmentation fix
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Initialize tracking structures
    num_concepts = autoencoder.dict_size
    
    # Use lists instead of heaps - much faster for appending!
    # We'll sort and trim to top-k at the end
    all_activations = defaultdict(list)  # concept_idx -> [(activation_value, img_tensor_or_idx, img_id), ...]
    concept_hashes = defaultdict(list) if use_hash_dedup else None
    
    # Use tensors for statistics - MUCH faster than defaultdict!
    activation_counts = torch.zeros(num_concepts, dtype=torch.long)
    activation_sums = torch.zeros(num_concepts, dtype=torch.float32)
    
    # TRACK MINIMUM VALUES: Track current minimum activation in top-k for each concept
    # Initialize to -inf so all values can be added initially
    concept_current_mins = torch.full((num_concepts,), float('-inf'), dtype=torch.float32)
    concept_list_sizes = torch.zeros(num_concepts, dtype=torch.long)
    
    print("\n=== PASS 1: Finding top-k images per concept (fast) ===")
    
    # Timing accumulators
    time_forward_pass = 0.0
    time_filtering = 0.0
    time_appending = 0.0
    time_total_batch = 0.0
    
    with torch.no_grad():
        for batch_idx, (batch_tensor, batch_ids) in enumerate(tqdm(dataloader, desc="Pass 1: Pooled activations", total=num_batches)):
            batch_start_time = time.time()
            
            batch_tensor_gpu = batch_tensor.to(args.device, non_blocking=True)
            
            # Get concept activations (pooled only - FAST!)
            forward_start = time.time()
            pooled_activations = model.get_aggregated_concept_activations(
                batch_tensor_gpu,
                construction_method=construction_method,
                use_threshold = True
            )
            time_forward_pass += time.time() - forward_start
            
            pooled_activations_cpu = pooled_activations.cpu()
            del pooled_activations, batch_tensor_gpu
            
            # Find activations above threshold
            above_threshold_mask = pooled_activations_cpu > threshold
            above_threshold_indices = torch.nonzero(above_threshold_mask, as_tuple=False)
            
            if above_threshold_indices.numel() > 0:
                above_threshold_values = pooled_activations_cpu[above_threshold_mask]
                batch_start_idx = batch_idx * batch_size
                
                # Debug: Print stats every 100 batches
                if batch_idx % 100 == 0 and batch_idx > 0:
                    print(f"\nBatch {batch_idx}: {above_threshold_indices.size(0)} activations above threshold")
                    filled_concepts = sum(1 for c_list in all_activations.values() if len(c_list) >= top_k)
                    print(f"Concepts with ≥{top_k} items: {filled_concepts}/{len(all_activations)}")
                
                # Vectorized approach - just append to lists!
                batch_positions = above_threshold_indices[:, 0]
                concept_indices = above_threshold_indices[:, 1]
                
                # VECTORIZED statistics update (like training.py) - NO FOR-LOOPS!
                # Use scatter_add for counts - super fast!
                activation_counts.scatter_add_(0, concept_indices, torch.ones_like(concept_indices, dtype=torch.long))
                
                # Use scatter_add for sums - super fast!
                activation_sums.scatter_add_(0, concept_indices, above_threshold_values)
                
                # GPU-ACCELERATED PRE-FILTERING: 100% VECTORIZED - NO FOR-LOOPS!
                # This is the KEY optimization - all filtering done with tensor operations!
                filter_start = time.time()
                unique_concepts = concept_indices.unique()
                
                # Compute max activation per concept in this batch (fully vectorized)
                batch_concept_max_values = torch.full((num_concepts,), float('-inf'))
                batch_concept_max_values.scatter_reduce_(0, concept_indices, above_threshold_values, reduce='amax', include_self=False)
                
                # Get current mins and sizes for unique concepts only (tensor indexing - fast!)
                unique_current_mins = concept_current_mins[unique_concepts]
                unique_list_sizes = concept_list_sizes[unique_concepts]
                unique_batch_maxes = batch_concept_max_values[unique_concepts]
                
                # VECTORIZED FILTER: concepts not full OR batch_max > current_min (pure tensor ops!)
                concepts_to_process_mask = (unique_list_sizes < top_k) | (unique_batch_maxes > unique_current_mins)
                concepts_to_process = unique_concepts[concepts_to_process_mask]
                time_filtering += time.time() - filter_start
                
                if batch_idx % 100 == 0 and batch_idx > 0:
                    print(f"  GPU filtering: {unique_concepts.size(0)} unique concepts → {concepts_to_process.size(0)} to process ({100*concepts_to_process.size(0)/unique_concepts.size(0):.1f}%)")
                
                # Now only iterate through filtered concepts (potentially 90% fewer!)
                append_start = time.time()
                for c_idx in concepts_to_process.tolist():
                    mask = concept_indices == c_idx
                    concept_positions = batch_positions[mask]
                    concept_values = above_threshold_values[mask]
                    
                    concept_list = all_activations[c_idx]
                    
                    # VERY aggressive trimming to prevent OOM
                    if len(concept_list) > top_k * 2:
                        if use_hash_dedup:
                            # Combine lists to sort them in parallel
                            # We zip the (list_item, hash) together
                            current_hashes = concept_hashes[c_idx]

                            # Handle rare case where lists are out of sync (e.g., from a past error)
                            if len(concept_list) != len(current_hashes):
                                print(f"\nWarning: Mismatch in list/hash size for c_idx {c_idx}. Re-hashing to recover.")
                                try:
                                    current_hashes = [
                                        imagehash.phash(get_pil_image_for_hashing(item[1], is_webdataset, dataset)) 
                                        for item in concept_list
                                    ]
                                    concept_hashes[c_idx] = current_hashes
                                except Exception as e:
                                    print(f"FATAL: Recovery failed for c_idx {c_idx}: {e}. Skipping trim.")
                                    continue # Skip trimming this concept to avoid crash

                            combined_list = list(zip(concept_list, current_hashes))

                            # Sort combined list by activation value (item[0][0])
                            combined_list.sort(reverse=True, key=lambda x: x[0][0])

                            # Trim to top-k
                            combined_list_trimmed = combined_list[:top_k]

                            # Un-zip the lists
                            all_activations[c_idx] = [item[0] for item in combined_list_trimmed]
                            concept_hashes[c_idx] = [item[1] for item in combined_list_trimmed]

                            # Update the local variable for the next loop
                            concept_list = all_activations[c_idx]
                        else:
                            concept_list.sort(reverse=True, key=lambda x: x[0])
                            all_activations[c_idx] = concept_list[:top_k]
                            concept_list = all_activations[c_idx]

                        # Update tracking: list was trimmed to exactly top_k
                        concept_list_sizes[c_idx] = top_k
                        concept_current_mins[c_idx] = concept_list[-1][0]
                    
                    # Only append if value might be in top-k
                    min_value = concept_list[-1][0] if len(concept_list) >= top_k else float('-inf')
                    
                    # OPTIMIZED: Filter first, then batch clone (avoid per-item clone overhead)
                    if is_webdataset:
                        # Build mask for values above min_value
                        above_min_mask = concept_values > min_value
                        positions_to_add = concept_positions[above_min_mask]
                        values_to_add = concept_values[above_min_mask]
                        
                        if len(positions_to_add) > 0:
                            # Batch clone: clone all needed tensors at once (MUCH faster!)
                            # CRITICAL: Move to CPU immediately to avoid GPU OOM!
                            img_tensors = batch_tensor[positions_to_add].clone().cpu()
                            
                            # # Append all at once
                            # for i, (batch_pos, activation_value) in enumerate(zip(positions_to_add.tolist(), values_to_add.tolist())):  # simple with posible duplicats
                            #     img_id = batch_ids[batch_pos] if isinstance(batch_ids, list) else str(batch_start_idx + batch_pos)
                            #     concept_list.append((activation_value, img_tensors[i], img_id))
                            for i, (batch_pos, activation_value) in enumerate(zip(positions_to_add.tolist(), values_to_add.tolist())):
                                
                                new_img_tensor = img_tensors[i]

                                img_id = batch_ids[batch_pos] if isinstance(batch_ids, list) else str(batch_start_idx + batch_pos)

                                if use_hash_dedup:
                                    # Get the list of hashes we've already stored for this concept
                                    seen_hashes = concept_hashes[c_idx]

                                    # 1. Hash the new candidate image
                                    try:
                                        pil_img = get_pil_image_for_hashing(new_img_tensor, is_webdataset=True, dataset_obj=None)
                                        current_hash = imagehash.phash(pil_img)
                                    except Exception as e:
                                        print(f"\nWarning: Could not hash image for concept {c_idx}. Skipping. Error: {e}")
                                        continue # Skip this image

                                    # 2. Check for duplicates
                                    is_duplicate = False
                                    for h in seen_hashes:
                                        if abs(current_hash - h) <= hash_threshold:
                                            is_duplicate = True
                                            break

                                    # 3. If it's a duplicate, skip it
                                    if is_duplicate:
                                        continue

                                    # 4. If not a duplicate, add it to the list
                                    concept_list.append((activation_value, new_img_tensor, img_id))

                                    # 5. Add its hash to the 'seen' list for this concept
                                    seen_hashes.append(current_hash)
                                else:
                                    concept_list.append((activation_value, new_img_tensor, img_id))
                    else:
                        # For non-webdataset, just store indices (no cloning needed)
                        above_min_mask = concept_values > min_value
                        positions_to_add = concept_positions[above_min_mask]
                        values_to_add = concept_values[above_min_mask]
                        
                        for batch_pos, activation_value in zip(positions_to_add.tolist(), values_to_add.tolist()):
                            actual_image_idx = batch_start_idx + batch_pos
                            concept_list.append((activation_value, actual_image_idx, None))
                    
                    # UPDATE TRACKING TENSORS after processing this concept
                    new_size = len(concept_list)
                    concept_list_sizes[c_idx] = new_size
                    if new_size >= top_k:
                        # List is sorted when we trim, but new items just appended
                        # Need to find actual minimum (could be old or new)
                        # Quick approximation: if list > top_k, we know min <= min_value
                        # For exact: would need to sort, but that's expensive
                        # Compromise: only update if we know it changed
                        if new_size == top_k:
                            # Just reached top_k, find the minimum
                            concept_current_mins[c_idx] = min(item[0] for item in concept_list)
                        elif new_size > top_k:
                            # Overfilled, need to track conservatively
                            # Use the old min_value as lower bound (may be outdated but safe)
                            concept_current_mins[c_idx] = min_value
                time_appending += time.time() - append_start
            
            # Track total batch time
            time_total_batch += time.time() - batch_start_time

            # Print detailed timing every 100 batches
            if batch_idx % 100 == 0 and batch_idx > 0:
                avg_forward = time_forward_pass / (batch_idx + 1)
                avg_filter = time_filtering / (batch_idx + 1)
                avg_append = time_appending / (batch_idx + 1)
                avg_total = time_total_batch / (batch_idx + 1)
                other_time = avg_total - avg_forward - avg_filter - avg_append
                
                print(f"\n=== Timing Breakdown (avg per batch) ===")
                print(f"  Forward pass (SAE):  {avg_forward:.3f}s ({100*avg_forward/avg_total:.1f}%)")
                print(f"  GPU filtering:       {avg_filter:.3f}s ({100*avg_filter/avg_total:.1f}%)")
                print(f"  Appending images:    {avg_append:.3f}s ({100*avg_append/avg_total:.1f}%)")
                print(f"  Other (I/O, etc):    {other_time:.3f}s ({100*other_time/avg_total:.1f}%)")
                print(f"  TOTAL:               {avg_total:.3f}s")
                
                est_remaining_time = avg_total * (num_batches - batch_idx) / 3600
                print(f"  Estimated remaining: {est_remaining_time:.2f} hours")
            
            # Aggressive memory cleanup
            del pooled_activations_cpu, batch_tensor
            if batch_idx % 5 == 0:  # More frequent cleanup
                torch.cuda.empty_cache()
                import gc
                gc.collect()
    
    print(f"\n=== PASS 1 COMPLETE ===")
    print(f"Found activations for {len(all_activations)} concepts")
    print(f"\n=== Pass 1 Total Time Breakdown ===")
    print(f"  Forward pass (SAE):  {time_forward_pass:.1f}s ({100*time_forward_pass/time_total_batch:.1f}%)")
    print(f"  GPU filtering:       {time_filtering:.1f}s ({100*time_filtering/time_total_batch:.1f}%)")
    print(f"  Appending images:    {time_appending:.1f}s ({100*time_appending/time_total_batch:.1f}%)")
    other_total = time_total_batch - time_forward_pass - time_filtering - time_appending
    print(f"  Other (I/O, etc):    {other_total:.1f}s ({100*other_total/time_total_batch:.1f}%)")
    print(f"  TOTAL:               {time_total_batch:.1f}s ({time_total_batch/3600:.2f} hours)")
    
    # Sort and trim to top-k for each concept
    print(f"\nSorting and selecting top-{top_k} per concept...")
    sort_start = time.time()
    top_activations = {}
    for concept_idx, acts in all_activations.items():
        # Sort by activation value (descending)
        acts.sort(reverse=True, key=lambda x: x[0])
        # Keep only top-k
        top_activations[concept_idx] = acts[:top_k]
    
    del all_activations
    if concept_hashes is not None:
        del concept_hashes
    sort_time = time.time() - sort_start
    print(f"Sorting took {sort_time:.1f}s")

    print(f"\n=== PASS 2: Computing activation maps for top-{top_k} images only ===")
    
    # Compute activation maps only for top-k images
    total_top_images = sum(len(acts) for acts in top_activations.values())
    print(f"Computing activation maps for {total_top_images} images total...")
    num_concepts_with_activations = len(top_activations)
    
    # Process in batches for efficiency (reduced batch size to avoid OOM)
    pass2_start = time.time()
    activation_map_batch_size = 8  # Small batch to avoid OOM with large SAE (524K dict_size)
    
    with torch.no_grad():  # Disable gradients for memory efficiency
        for concept_idx in tqdm(top_activations, desc="Pass 2: Activation maps", total=num_concepts_with_activations):
            concept_acts = top_activations[concept_idx]
            
            # Collect images for this concept
            if is_webdataset:
                images = [item[1] for item in concept_acts]  # img_tensor (index 1)
            else:
                images = [dataset[item[1]][0] for item in concept_acts]  # load by image_idx (index 1)
            
            # Process in batches
            for batch_start in range(0, len(images), activation_map_batch_size):
                batch_end = min(batch_start + activation_map_batch_size, len(images))
                img_batch = torch.stack(images[batch_start:batch_end]).to(args.device)
                
                # Compute activation maps
                act_maps = model.get_concept_activation_map(
                    img_batch,
                    use_threshold=True,
                )
                
                # Extract this concept's activation maps (move to CPU immediately)
                concept_act_maps = act_maps[:, concept_idx].cpu()
                
                # Attach to the items
                for i in range(batch_end - batch_start):
                    item_idx = batch_start + i
                    # Add activation map to the tuple: (act_val, img_tensor_or_idx, img_id_or_None)
                    if is_webdataset:
                        act_val, img_tensor, img_id = concept_acts[item_idx]
                        concept_acts[item_idx] = (act_val, img_tensor, img_id, concept_act_maps[i])
                    else:
                        act_val, image_idx, _ = concept_acts[item_idx]
                        concept_acts[item_idx] = (act_val, image_idx, None, concept_act_maps[i])
                
                # Explicitly delete GPU tensors and clear cache
                del img_batch, act_maps, concept_act_maps
                torch.cuda.empty_cache()
            
            # Clear the images list to free CPU memory
            del images
    
    pass2_time = time.time() - pass2_start
    print(f"\n=== PASS 2 COMPLETE ===")
    print(f"All activation maps computed in {pass2_time:.1f}s ({pass2_time/60:.1f} minutes)")
    
    # VECTORIZED statistics computation (like training.py) - NO FOR-LOOPS!
    stats_start = time.time()
    activation_frequency = activation_counts.float() / total_images
    
    # Mean activation magnitude: sum / count (avoid division by zero)
    mean_activation_magnitude = torch.zeros(num_concepts)
    nonzero_mask = activation_counts > 0
    mean_activation_magnitude[nonzero_mask] = activation_sums[nonzero_mask] / activation_counts[nonzero_mask].float()
    stats_time = time.time() - stats_start
    
    print(f"\nActivation Statistics (computed in {stats_time:.3f}s):")
    print(f"  Mean activation frequency: {activation_frequency.mean().item():.6f}")
    print(f"  Max activation frequency: {activation_frequency.max().item():.6f}")
    print(f"  Min activation frequency: {activation_frequency.min().item():.6f}")
    print(f"  Mean activation magnitude (when activated): {mean_activation_magnitude[nonzero_mask].mean().item():.6f}")
    
    # Save images and activation maps to subfolders
    print(f"\nSaving top-{top_k} images and activation maps per concept to {output_dir}...")
    save_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    if is_webdataset:
        # For webdataset, images are already stored as tensors
        for concept_idx in tqdm(top_activations, desc="Saving images"):
            concept_dir = osp.join(output_dir, f"concept_{concept_idx:04d}")
            os.makedirs(concept_dir, exist_ok=True)
            
            for rank, (activation_value, img_tensor, img_id, act_map) in enumerate(top_activations[concept_idx]):
                try:
                    img_tensor_save = img_tensor.cpu()
                    img_np = img_tensor_save.permute(1, 2, 0).numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_np)
                    
                    img_filename = f"rank_{rank:02d}_id_{img_id}_act_{activation_value:.4f}.jpg"
                    img_pil.save(osp.join(concept_dir, img_filename), quality=95)
                    
                    # Save the activation map visualization (whitened image)
                    save_activation_map(img_np, act_map, osp.join(concept_dir, f"rank_{rank:02d}_id_{img_id}_actmap.jpg"))
                    
                    # Save the raw activation map tensor (14x14)
                    torch.save(act_map, osp.join(concept_dir, f"rank_{rank:02d}_id_{img_id}_actmap.pt"))
                    
                except Exception as e:
                    print(f"Error saving image {img_id} for concept {concept_idx}: {e}")
                    continue
    else:
        # For indexable datasets, retrieve images by index
        for concept_idx in tqdm(top_activations, desc="Saving images"):
            concept_dir = osp.join(output_dir, f"concept_{concept_idx:04d}")
            os.makedirs(concept_dir, exist_ok=True)
            
            for rank, (activation_value, image_idx, _, act_map) in enumerate(top_activations[concept_idx]):
                try:
                    # Get image from dataset
                    img_tensor = dataset[image_idx][0].cpu()
                    img_np = img_tensor.permute(1, 2, 0).numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_np)
                    
                    img_filename = f"rank_{rank:02d}_imgidx_{image_idx}_act_{activation_value:.4f}.jpg"
                    img_pil.save(osp.join(concept_dir, img_filename), quality=95)
                    
                    # Save the activation map visualization (whitened image)
                    save_activation_map(img_np, act_map, osp.join(concept_dir, f"rank_{rank:02d}_imgidx_{image_idx}_actmap.jpg"))
                    
                    # Save the raw activation map tensor (14x14)
                    torch.save(act_map, osp.join(concept_dir, f"rank_{rank:02d}_imgidx_{image_idx}_actmap.pt"))
                    
                except Exception as e:
                    print(f"Error saving image {image_idx} for concept {concept_idx}: {e}")
                    continue
    
    # Save statistics (convert top_activations to a serializable format)
    if is_webdataset:
        # For webdataset, don't save the image tensors (too large), just metadata
        top_activations_meta = {
            concept_idx: [(act_val, img_id) for act_val, _, img_id, _ in acts]
            for concept_idx, acts in top_activations.items()
        }
    else:
        top_activations_meta = {
            concept_idx: [(act_val, img_idx) for act_val, img_idx, _, _ in acts]
            for concept_idx, acts in top_activations.items()
        }
    
    stats = {
        'activation_frequency': activation_frequency,
        'mean_activation_magnitude': mean_activation_magnitude,
        'top_activations': top_activations_meta,
        'total_images': total_images,
        'threshold': threshold,
        'construction_method': construction_method,
    }
    
    stats_path = osp.join(output_dir, 'activation_statistics.pt')
    torch.save(stats, stats_path)
    save_time = time.time() - save_start
    
    print(f"\nSaved statistics to {stats_path}")
    print(f"Image saving took {save_time:.1f}s ({save_time/60:.1f} minutes)")
    
    # Final summary
    total_time = time_total_batch + sort_time + pass2_time + stats_time + save_time
    print(f"\n{'='*60}")
    print(f"=== FINAL TIMING SUMMARY ===")
    print(f"{'='*60}")
    print(f"Pass 1 (find top-k):        {time_total_batch:.1f}s ({time_total_batch/3600:.2f}h) - {100*time_total_batch/total_time:.1f}%")
    print(f"  - Forward pass (SAE):     {time_forward_pass:.1f}s - {100*time_forward_pass/total_time:.1f}%")
    print(f"  - GPU filtering:          {time_filtering:.1f}s - {100*time_filtering/total_time:.1f}%")
    print(f"  - Appending images:       {time_appending:.1f}s - {100*time_appending/total_time:.1f}%")
    print(f"Sorting:                    {sort_time:.1f}s ({sort_time/60:.1f}m) - {100*sort_time/total_time:.1f}%")
    print(f"Pass 2 (activation maps):   {pass2_time:.1f}s ({pass2_time/60:.1f}m) - {100*pass2_time/total_time:.1f}%")
    print(f"Statistics computation:     {stats_time:.1f}s - {100*stats_time/total_time:.1f}%")
    print(f"Saving images to disk:      {save_time:.1f}s ({save_time/60:.1f}m) - {100*save_time/total_time:.1f}%")
    print(f"{'='*60}")
    print(f"TOTAL TIME:                 {total_time:.1f}s ({total_time/3600:.2f} hours)")
    print(f"{'='*60}")
    
    return stats


def get_whitened_img(img, act_tensor, multiply_factor=3, power=2):
    """
    Create a whitened visualization where low-activation areas are pushed toward white.

    Args:
        img: HxWx3 uint8 numpy array or PIL image
        act_tensor: torch tensor of shape (H_act, W_act)
    """
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img

    # Make localization sharper by emphasizing high activations.
    act_tensor = act_tensor.float() ** power
    act_tensor = F.interpolate(
        act_tensor.unsqueeze(0).unsqueeze(0),
        size=img_np.shape[:2],
        mode='bilinear',
        align_corners=False,
    ).squeeze()

    max_val = act_tensor.max()
    if max_val > 0:
        act_tensor = act_tensor / max_val

    act_tensor = act_tensor * multiply_factor
    act = act_tensor.unsqueeze(2).repeat(1, 1, 3).numpy().clip(0, 1)
    actmap_np = (act * img_np + (1 - act) * 255).astype(np.uint8)
    return Image.fromarray(actmap_np)


def save_activation_map(img_np, act_map, save_path):
    """
    Save a whitened activation visualization image.
    
    Args:
        img_np: input image as HxWx3 uint8 numpy array
        act_map: torch tensor of shape (H, W), e.g., (14, 14)
        save_path: path to save the heatmap image
    """
    whitened = get_whitened_img(img_np, act_map)
    whitened.save(save_path, quality=95)


def main():
    parser = arg_parser.get_default_parser()
    parser.add_argument("--threshold", type=float, default=0.1, 
                       help="Minimum activation threshold")
    parser.add_argument("--top_k", type=int, default=10,
                       help="Number of top images to save per concept")
    
    args = parser.parse_args()
    common_init(args, disable_make_dirs=True)
    
    # Setup paths
    base_dir = osp.join(
        config.analysis_dir,
        'task_agnosticity',
        args.sae_dataset,
        args.img_enc_name_for_saving,
        args.config_name
    )
    
    output_dir = osp.join(base_dir, f"{args.probe_dataset}_{args.probe_split}_top_imgs_{args.top_k}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Configuration:")
    print(f"  SAE dataset: {args.sae_dataset}")
    print(f"  Probe dataset: {args.probe_dataset}")
    print(f"  Model: {args.img_enc_name}")
    print(f"  Config: {args.config_name}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Construction method: {args.probe_aggregation_method}")
    print(f"  Output directory: {output_dir}")
    print(f"  Probe Split: {args.probe_split}")
    
    # Load model and autoencoder
    print("\nLoading model and autoencoder...")
    model, preprocess = get_img_model(args)
    base_path = os.path.join(args.save_dir_sae_ckpts["img"], args.save_suffix) + args.config_name + "/" + 'trainer_0'
    autoencoder, ae_cfg = load_dictionary(base_path, args.device)
    cfm = CFM(
            feature_extractor=model,
            autoencoder=autoencoder,
            device=args.device
        )

    # NOTE: DataParallel doesn't work well here because the model internally calls
    # autoencoder.encode() which DataParallel can't parallelize properly.
    # Instead, we optimize with larger batch sizes and better I/O.
    print(f"\nUsing single GPU: {args.device}")
    print("Note: Multi-GPU not used - model architecture doesn't support DataParallel")
    print("      Speed optimized via batch size and I/O parallelization instead")
    
    # Load dataset
    print(f"\nLoading {args.probe_dataset} dataset...")
    dataset = get_probe_dataset(
        args.probe_dataset, 
        args.probe_split, 
        args.probe_dataset_root_dir, 
        preprocess_fn=preprocess
    )
    
    # Compute activations and save images
    stats = compute_activations_and_save_images(
        args=args,
        dataset=dataset,
        model=cfm,
        autoencoder=autoencoder,
        output_dir=output_dir,
        threshold=args.threshold,
        top_k=args.top_k,
        construction_method=args.probe_aggregation_method
    )
    
    print("\nDone!")
    print(f"Images saved to: {output_dir}")
    print(f"Statistics saved to: {osp.join(output_dir, 'activation_statistics.pt')}")


if __name__ == "__main__":
    main()
