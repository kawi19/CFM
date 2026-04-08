import os

from cfm.data_utils import cc12m
import torch

import os.path as osp

from cfm.utils import common_init, get_img_model, get_probe_dataset
from cfm import arg_parser
import numpy as np
import gc

import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class FetchTokens:
    def __init__(self, args=None):
        self.model, self.preprocess = get_img_model(args)
        self.model.eval()  # Set model to evaluation mode
        self.args = args
    
    def get_dataset_size_and_shape(self, loader):
        """Get total dataset size and activation shape for memmap initialization"""
        total_samples = 0
        activation_shape = None
        with torch.no_grad():
            for (inputs, idx) in loader:
                if activation_shape is None:
                    # Get shape from first batch
                    sample_activation = self.extract_all_tokens(inputs[:1].to(self.args.device))
                    activation_shape = sample_activation.shape[1:]  # Remove batch dimension
                break  # Only need first batch for shape
        if self.args.probe_dataset == "cc12m":
            total_samples =  10968539 
            self.total_samples = total_samples
        else:
            total_samples = loader.dataset.__len__()
            self.total_samples = total_samples
        return total_samples, activation_shape

    def dump_idxs(self):
        """ split the train indices into train and train_val;  for training and validation on the probe dataset"""
        assert (self.args.probe_split == "train")
        train_dataset = get_probe_dataset(self.args.probe_dataset, self.args.probe_split, self.args.probe_dataset_root_dir, self.preprocess)
        total_samples = self.total_samples  # if self.args.probe_dataset == "cc12m" else train_dataset.__len__()
        randperm = torch.randperm(total_samples)
        train_prop = 0.9
        train_num = int(train_prop * total_samples)
        train_idxs = randperm[:train_num]
        train_val_idxs = randperm[train_num:]

        assert (train_idxs.shape[0]+train_val_idxs.shape[0] == randperm.shape[0] == total_samples)
        os.makedirs(self.args.probe_split_idxs_dir['img'], exist_ok=True)
        torch.save(train_idxs, os.path.join(
            self.args.probe_split_idxs_dir['img'], "train_idxs.pth"))
        torch.save(train_val_idxs, os.path.join(
            self.args.probe_split_idxs_dir['img'], "train_val_idxs.pth"))
        
    def extract_all_tokens(self, images, num_tokens=None):
        """
        This function extracts dense features from the model. "num_tokens" if given defines the number of tokens to subsample per image. None means no subsampling.
        """
        if self.args.img_enc_name.startswith("dinoclip"):
            print("images shape:", images.shape)
            x = self.model.get_pooled_feats(images) 
            print("x shape after get_pooled_feats:", x.shape)
            x_p = x.permute(0,2,3,1)
            x_p = x_p.reshape(x_p.shape[0],x_p.shape[1]*x_p.shape[2],x_p.shape[3])
            if num_tokens is not None:
                # randomly sample subset_num tokens out of all tokens per image
                batch_dim, token_dim, embed_dim = x_p.shape
                
                # Generate random noise for each token in the batch
                noise = torch.rand(batch_dim, token_dim, device=x.device)
                
                # This is an efficient way to get a random permutation without replacement
                _, idx = torch.topk(noise, num_tokens, dim=1)

                # The gather operation remains identical
                x_p = x_p.gather(1, idx.unsqueeze(-1).expand(-1, -1, embed_dim))
            return x_p
        else:
            raise NotImplementedError("extract_all_tokens is only implemented for dinoclip models.")

    def get_probe_out_memmap(self, loader, output_path, labels_path, save_labels=True, new_num_tokens = None ):
        """Process data and save directly to memmap files"""
        # First pass: get total size and shape
        total_samples, activation_shape = self.get_dataset_size_and_shape(loader)
        
        save_labels = self.args.probe_dataset != "cc12m"

        print(f"Total samples: {total_samples}, Activation shape: {activation_shape}")
        # estimate how many tokens to save per image
        activation_shape = list(activation_shape)
        if new_num_tokens!= None:
            activation_shape[0] = new_num_tokens # subsample tokens per image
        activation_shape = tuple(activation_shape)
        # Create memmap files
        full_shape = (total_samples,) + activation_shape
        
        # # Create memmap arrays
        output_memmap = np.memmap(
            output_path, 
            dtype=np.float32, 
            mode='w+', 
            shape=full_shape
        )
        labels_memmap = "does not exist"
        if save_labels:
            labels_memmap = np.memmap(
                labels_path,
                dtype=np.int64,
                mode='w+', 
                shape=(total_samples,)
            )
        
        # Save metadata for both files
        self.save_memmap_metadata(full_shape, np.float32, output_path)
        if save_labels:
            self.save_memmap_metadata((total_samples,), np.int64, labels_path)
        
        # Second pass: fill the memmap arrays
        current_idx = 0
        torch.cuda.empty_cache()
        gc.collect()
        flush_interval = 5  # Flush every 5 batches to avoid memory buildup
        # new_num_tokens = None  # save all tokens for now
        with torch.no_grad():
            for batch_num, (inputs, idx) in enumerate(tqdm(loader, desc="Processing batches", unit="batch", total=self.total_samples // self.args.batch_size)):
                inputs = inputs.to(self.args.device)
                
                # Extract activations
                batch_output = self.extract_all_tokens(inputs, num_tokens=new_num_tokens).detach().cpu().numpy() # can adjust how many tokens are saved with subset_num=new_num_tokens
                print("batch_output shape:", batch_output.shape)
                batch_size = batch_output.shape[0]
                # Write to memmap
                output_memmap[current_idx:current_idx + batch_size] = batch_output
                if save_labels:
                    labels_memmap[current_idx:current_idx + batch_size] = idx.numpy()
                current_idx += batch_size

                # Flush periodically to avoid memory buildup
                if (batch_num + 1) % flush_interval == 0:
                    output_memmap.flush()
                    if save_labels:
                        labels_memmap.flush()
                    torch.cuda.empty_cache()
                    gc.collect()

                if current_idx % 1000 == 0:  # Progress update
                    print(f"Processed {current_idx}/{total_samples} samples")
                # break # TODO remove 
        print(f"Finished processing. Total samples processed: {current_idx}")
        # Ensure data is written to disk
        output_memmap.flush()
        if save_labels:
            labels_memmap.flush()

        if current_idx < total_samples:
            print(f"Trimming memmap from {total_samples} to {current_idx} samples")
            total_samples = current_idx  # Update total samples to actual count
            self.total_samples = total_samples
            row_bytes = output_memmap[0].nbytes  # bytes per sample
            new_size_bytes = current_idx * row_bytes

            # 1. Flush anything still in cache
            output_memmap.flush()

            # 2. Truncate the underlying file
            with open(output_path, "r+b") as f:
                f.truncate(new_size_bytes)

            # 3. Update metadata (so future reads know it's smaller)
            self.save_memmap_metadata(
                (current_idx,) + activation_shape, np.float32, output_path
            )
        return output_memmap, labels_memmap

    def save_probe_tokens_memmap(self, probe_dataset, do_val = False):
        """Save tokens using memmap for memory efficiency"""
        # Create output directory
        os.makedirs(self.args.probe_data_dir_activations["img"], exist_ok=True)
        # Process train split
        train_dataset = get_probe_dataset(
            probe_dataset, 'train', self.args.probe_dataset_root_dir, self.preprocess)
        if probe_dataset == "cc12m":
            train_loader = cc12m.get_dataloader(train_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=12)
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.args.batch_size, shuffle=False)
        save_labels = probe_dataset != "cc12m"

        # Define memmap file paths
        train_output_path = osp.join(self.args.probe_data_dir_activations["img"], "train_full.dat")
        train_labels_path = osp.join(self.args.probe_split_idxs_dir['img'], "all_labels_train.dat")
        
        # Save full train data to memmap
        print("Processing train data...")
        train_output_memmap, train_labels_memmap = self.get_probe_out_memmap(
            train_loader, train_output_path, train_labels_path, save_labels=save_labels)
        print("Data saved successfully using memmap!")
        # loading idx
        if not osp.exists(os.path.join(self.args.probe_split_idxs_dir['img'], "train_idxs.pth")):
            print(f"\n Labels do not already exist! \n")
            self.dump_idxs()

        # Load split indices
        train_idxs = torch.load(os.path.join(
            self.args.probe_split_idxs_dir['img'], "train_idxs.pth"))
        train_val_idxs = torch.load(os.path.join(
            self.args.probe_split_idxs_dir['img'], "train_val_idxs.pth"))
        
        # print("Data saved successfully using memmap!")
        print("Loading full train data from memmap...")
        train_output_memmap, train_labels_memmap = self.load_memmap_data(split='train_full', do_labels=save_labels)
        print("now processing train split...")
        
        # Create split memmaps
        train_split_path = osp.join(self.args.probe_data_dir_activations["img"], "train.dat")
        train_val_split_path = osp.join(self.args.probe_data_dir_activations["img"], "train_val.dat")
        train_labels_split_path = osp.join(self.args.probe_split_idxs_dir['img'], "labels_train.dat")
        train_val_labels_split_path = osp.join(self.args.probe_split_idxs_dir['img'], "labels_train_val.dat")
        
        # Create train split memmap
        train_split_memmap = np.memmap(
            train_split_path,
            dtype=np.float32,
            mode='w+',
            shape=(len(train_idxs),) + train_output_memmap.shape[1:]
        )

        train_val_split_memmap = np.memmap(
            train_val_split_path,
            dtype=np.float32,
            mode='w+',
            shape=(len(train_val_idxs),) + train_output_memmap.shape[1:]
        )

        # One can tweak these depending on node RAM and I/O speed
        MAX_WORKERS = 8  # number of parallel I/O threads
        CHUNK_SIZE = 20_000  # number of rows per chunk

        def sorted_copy_chunk(src, dst, sorted_indices, dst_positions, start, end, desc):
            """Copy one chunk safely."""
            idx_chunk = sorted_indices[start:end]
            pos_chunk = dst_positions[start:end]

            block = np.empty((len(idx_chunk),) + src.shape[1:], dtype=np.float32)
            for j, src_idx in enumerate(idx_chunk):
                block[j] = src[src_idx]
            dst[pos_chunk] = block
            return end


        def sorted_copy_parallel(source_memmap, target_memmap, indices, desc, n_workers=MAX_WORKERS):
            """Parallel sorted copy for memmaps with safe partitioning."""
            indices = np.asarray(indices)
            sorted_order = np.argsort(indices)
            sorted_indices = indices[sorted_order]

            n = len(sorted_indices)
            step = CHUNK_SIZE
            starts = list(range(0, n, step))
            ends = [min(s + step, n) for s in starts]

            print(f"[{desc}] Using {n_workers} workers, chunk size={CHUNK_SIZE:,}, total chunks={len(starts)}")

            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = []
                for i, (start, end) in enumerate(zip(starts, ends)):
                    futures.append(
                        ex.submit(
                            sorted_copy_chunk,
                            source_memmap,
                            target_memmap,
                            sorted_indices,
                            sorted_order,
                            start,
                            end,
                            f"{desc} [chunk {i+1}/{len(starts)}]"
                        )
                    )

                for f in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="chunk"):
                    f.result()

            target_memmap.flush()


        # ----------- TRAIN SPLIT -----------
        sorted_copy_parallel(
            source_memmap=train_output_memmap,
            target_memmap=train_split_memmap,
            indices=train_idxs,
            desc="Copying train split (sorted)",
            n_workers=MAX_WORKERS,
        )

        self.save_memmap_metadata(
            (len(train_idxs),) + train_output_memmap.shape[1:], 
            np.float32, 
            train_split_path
        )

        # ----------- TRAIN_VAL SPLIT -----------
        sorted_copy_parallel(
            source_memmap=train_output_memmap,
            target_memmap=train_val_split_memmap,
            indices=train_val_idxs,
            desc="Copying train_val split (sorted)",
            n_workers=MAX_WORKERS,
        )

        self.save_memmap_metadata(
            (len(train_val_idxs),) + train_output_memmap.shape[1:], 
            np.float32, 
            train_val_split_path
        )
        if save_labels: # this is false for cc12m
            # Create label split memmaps
            train_labels_split_memmap = np.memmap(
                train_labels_split_path,
                dtype=np.int64,
                mode='w+',
                shape=(len(train_idxs),)
            )
            train_labels_split_memmap[:] = train_labels_memmap[train_idxs]
            train_labels_split_memmap.flush()
            
            # Save metadata for train labels
            self.save_memmap_metadata((len(train_idxs),), np.int64, train_labels_split_path)
            
            train_val_labels_split_memmap = np.memmap(
                train_val_labels_split_path,
                dtype=np.int64,
                mode='w+',
                shape=(len(train_val_idxs),)
            )
            train_val_labels_split_memmap[:] = train_labels_memmap[train_val_idxs]
            train_val_labels_split_memmap.flush()
            
            # Save metadata for train_val labels
            self.save_memmap_metadata((len(train_val_idxs),), np.int64, train_val_labels_split_path)
        
        # Explicitly delete references to free memory
        if save_labels:
            del train_split_memmap, train_val_split_memmap
            del train_labels_split_memmap, train_val_labels_split_memmap
        
            # Clean up full arrays (they're still on disk)
            del train_output_memmap, train_labels_memmap
        else:
            # del train_split_memmap
            del train_val_split_memmap
            del train_output_memmap
        
        # Process val split
        if do_val:
            print("now processing val split...")
            val_dataset = get_probe_dataset(
                self.args.probe_dataset, 'val', self.args.probe_dataset_root_dir, self.preprocess)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.args.batch_size, shuffle=False)
            
            val_output_path = osp.join(self.args.probe_data_dir_activations["img"], "val.dat")
            val_labels_path = osp.join(self.args.probe_split_idxs_dir['img'], "all_labels_val.dat")
            
            print("Processing val data...")
            val_output_memmap, val_labels_memmap = self.get_probe_out_memmap(
                val_loader, val_output_path, val_labels_path)
        if os.path.exists(train_output_path):
            os.remove(train_output_path)
            print(f"Deleted full training memmap at: {train_output_path}")
            # print(f"{train_output_path} retained for future use.")
        
        print("Data saved successfully using memmap!")

    # Save metadata for easier loading
    def save_memmap_metadata(self, shape, dtype, path):
        """Save metadata for memmap files"""
        metadata = {
            'shape': shape,
            'dtype': str(dtype)
        }
        import pickle
        with open(path + '.meta', 'wb') as f:
            pickle.dump(metadata, f)
    
    def load_memmap_data(self, split='train', do_labels = True):
        """Load data from memmap files using metadata"""
        if split == 'train':
            data_path = osp.join(self.args.probe_data_dir_activations["img"], "train.dat")
            labels_path = osp.join(self.args.probe_split_idxs_dir['img'], "labels_train.dat")
        elif split == 'train_val':
            data_path = osp.join(self.args.probe_data_dir_activations["img"], "train_val.dat")
            labels_path = osp.join(self.args.probe_split_idxs_dir['img'], "labels_train_val.dat")
        elif split == 'val':
            data_path = osp.join(self.args.probe_data_dir_activations["img"], "val.dat")
            labels_path = osp.join(self.args.probe_split_idxs_dir['img'], "all_labels_val.dat")
        elif split == 'train_full':
            data_path = osp.join(self.args.probe_data_dir_activations["img"], "train_full.dat")
            labels_path = osp.join(self.args.probe_split_idxs_dir['img'], "all_labels_train.dat")
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Load using metadata
        data_memmap = self.load_memmap_with_metadata(data_path)
        if do_labels:
            labels_memmap = self.load_memmap_with_metadata(labels_path)
        else:
            labels_memmap = "no labels"
        
        return data_memmap, labels_memmap
    
    def load_memmap_with_metadata(self, data_path):
        """Load memmap with metadata"""
        import pickle
        with open(data_path + '.meta', 'rb') as f:
            metadata = pickle.load(f)
        
        # Convert string dtype back to numpy dtype
        dtype_str = metadata['dtype']
        if dtype_str == "<class 'numpy.float32'>" or dtype_str == "float32":
            dtype = np.float32
        elif dtype_str == "<class 'numpy.int64'>" or dtype_str == "int64":
            dtype = np.int64
        elif dtype_str == "<class 'numpy.float64'>" or dtype_str == "float64":
            dtype = np.float64
        elif dtype_str == "<class 'numpy.int32'>" or dtype_str == "int32":
            dtype = np.int32
        else:
            # Fallback: try to convert string to numpy dtype
            try:
                # Handle cases like 'numpy.float32' or just 'float32'
                if 'numpy.' in dtype_str:
                    dtype_name = dtype_str.split('.')[-1].rstrip("'>")
                else:
                    dtype_name = dtype_str
                dtype = getattr(np, dtype_name)
            except (AttributeError, ValueError):
                raise ValueError(f"Unknown dtype in metadata: {dtype_str}")
        
        return np.memmap(
            data_path, 
            dtype=dtype, 
            mode='r',
            shape=tuple(metadata['shape'])
        )

if __name__ == '__main__':
    # Run this file if you want to save dense features F+
    parser = arg_parser.get_default_parser()
    parser.add_argument("--batch_size", type=int, default=1024)  # original 4096 for base models 1024 for Large
    args = parser.parse_args()
    common_init(args)
    print(vars(args))

    fetch_act = FetchTokens(args)
    # to save probe_tokens
    if args.probe_dataset in ["cc12m"]:
        do_val = False
    else:
        do_val = True
    fetch_act.save_probe_tokens_memmap(args.probe_dataset, do_val=do_val)
    # fetch_act.save_probe_tokens_memmap_only_val(args.probe_dataset)
