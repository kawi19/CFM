import os  
import torch
import os

from tqdm import tqdm
from dictionary_learning.utils import load_dictionary
import pickle 
import numpy as np
from torch.utils.data import DataLoader
from cfm import arg_parser
from cfm.utils import common_init
from cfm.data_utils.activation_store.tensor_store import TensorActivationStore

def save_concept_strengths(args, dir_path):
    if args.sae_type!= None: 
        if args.sae_type == "matryoshka":
            base_path = os.path.join(args.save_dir_sae_ckpts['img'], args.save_suffix) + args.config_name + "/" + 'trainer_0'
            print("loading autoencoder")
            autoencoder, ae_config =  load_dictionary(base_path, args.device)
        else:
            base_path = os.path.join(args.save_dir_sae_ckpts['img'],args.save_suffix) + "trainer_0"
            print("loading autoencoder")
            autoencoder, config =  load_dictionary(base_path, args.device)
    else:
        raise NotImplementedError
    autoencoder.eval()
    if args.sae_dataset == "imagenet" or args.sae_dataset == "places365" or args.sae_dataset == "cc12m": # only large datasets are saved as memmaps
        is_memmap = True
    else:
        is_memmap = False
    if not(is_memmap):
        if args.probe_split!= "train":
            raise NotImplementedError
        autoencoder_input_dim:int = args.autoencoder_input_dim_dict[args.ae_input_dim_dict_key['img']]
        fnames = os.listdir(args.data_dir_activations['img']) # had to add .pth to the path
        embedding_dir = args.data_dir_activations['img']
        print(f"Getting fnames from {args.data_dir_activations['img']}")
        train_fnames = []
        train_val_fnames = []
        for fname in fnames:
            if fname.startswith(f"train_val"):
                train_val_fnames.append(os.path.join(
                    os.path.abspath(embedding_dir), fname))
            elif fname.startswith(f"train"):
                train_fnames.append(os.path.join(
                    os.path.abspath(embedding_dir), fname))
                
        if args.val_freq == 0:   
            train_fnames = train_fnames + train_val_fnames
            train_val_fnames = None

        print(f"train_fnames: {train_fnames}")
        print(f"train_val_fnames: {train_val_fnames}")



        def get_activation_store(activation_fname):
            activations = torch.load(activation_fname, weights_only=True)
            if len(activations[0].shape) == 2:    # for token activations, these are stored in 3d
                # activations = activations[:,:, :]   # excluding the class token   # not excluding the class token anymore
                activations = activations.reshape(-1, autoencoder_input_dim)
            print("activations shape:")
            print(activations.shape)
            activation_store = TensorActivationStore(
                activations.shape[0], autoencoder_input_dim, 1)
            activation_store.empty()
            activation_store.extend(activations, component_idx=0)
            return activation_store

        num_train_pieces = len(train_fnames)
        train_piece_idx = 0
        train_order = torch.randperm(num_train_pieces)
        train_dataset = get_activation_store(train_fnames[train_order[train_piece_idx]])
        number_of_samples = len(train_dataset)
        print("training samples:", number_of_samples)

        if train_val_fnames!= None:
            num_trainval_pieces = len(train_val_fnames)
            trainval_piece_idx = 0
            trainval_order = torch.randperm(num_trainval_pieces)
            trainval_dataset = get_activation_store(train_val_fnames[trainval_order[trainval_piece_idx]])
            number_of_trainval_samples = len(trainval_dataset)
            print("train val samples:", number_of_trainval_samples)

        def custom_collate_fn(batch):
            # Convert the batch to a tensor and squeeze the second dimension
            batch = torch.stack(batch)  
            return batch.squeeze(1)


        loader = DataLoader(train_dataset, batch_size=args.train_sae_bs, shuffle=True, collate_fn=custom_collate_fn)
    else:
        def load_memmap_with_metadata(data_path):
            """Load memmap with metadata"""

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

        class ChunkedSequentialReader:
            """Reads data in large chunks sequentially"""
            def __init__(self, path, autoencoder_input_dim, batch_size=256, chunk_size=8192):
                self.memmap = load_memmap_with_metadata(path)
                self.memmap = self.memmap.reshape(-1, autoencoder_input_dim)
                self.batch_size = batch_size
                self.chunk_size = chunk_size
                self.total_samples = self.memmap.shape[0]
                print(f"Dataset shape: {self.memmap.shape}")
                
            def __iter__(self):
                for start_idx in range(0, self.total_samples, self.chunk_size):
                    end_idx = min(start_idx + self.chunk_size, self.total_samples)
                    
                    # Read a large chunk into memory
                    chunk = self.memmap[start_idx:end_idx]
                    chunk_tensor = torch.tensor(chunk, dtype=torch.float32)
                    
                    # Yield batches from this chunk
                    for batch_start in range(0, len(chunk_tensor), self.batch_size):
                        batch_end = min(batch_start + self.batch_size, len(chunk_tensor))
                        yield chunk_tensor[batch_start:batch_end]

        # Chunked reader (faster)
        def create_chunked_reader(args):
            train_split_path = os.path.join(args.probe_data_dir_activations["img"], "train.dat")
            autoencoder_input_dim = args.autoencoder_input_dim_dict[args.ae_input_dim_dict_key['img']]
            
            return ChunkedSequentialReader(
                train_split_path, 
                autoencoder_input_dim,
                batch_size=16384,
                chunk_size=65536  # Adjust based on memory
            )
        loader = create_chunked_reader(args)
        # print("Length of loader:", len(loader))
        print("Dataset length:", loader.total_samples)

    print("starting to compute coocurance matrix and top-k activations")
    num_concepts = autoencoder.dict_size
    activation_threshold = 0.001
    top_k = 30
    print(f"num_concepts: {num_concepts}, activation_threshold: {activation_threshold}, top_k: {top_k}")

    C = torch.zeros((num_concepts, num_concepts), dtype=torch.float32, device=args.device)
    D = torch.zeros((num_concepts, num_concepts), dtype=torch.float32, device=args.device)
    topk_values = torch.full((num_concepts, top_k), float('-inf'), device=args.device)
    topk_patch_activations = torch.zeros((num_concepts, top_k, num_concepts), dtype=torch.float32, device=args.device)
    feature_frequencies = torch.zeros((num_concepts, 1), dtype=torch.float32, device=args.device)
    feature_sums = torch.zeros((num_concepts, 1), dtype=torch.float32, device=args.device)
    with torch.no_grad():
        for features in tqdm(loader):
            features = features.to(args.device) # [0] for non memmap
            if args.sae_type!= None:
                if args.sae_type== 'topk':
                    concepts, _, _, _ =  autoencoder.encode(features, return_topk = True, use_threshold = True)
                else:
                    concepts=  autoencoder.encode(features)
            else: 
                concepts, reconstructions = autoencoder(features)
            concepts= concepts.squeeze()       
            # compute running coocurance matrix
            concepts = concepts.view(-1, num_concepts) # should now be of size [batch_tokens, concepts]
            concepts_t = concepts.T # now of size [concepts, batch_tokens]
            A_sub = (concepts_t > activation_threshold).float()
            C_sub = torch.matmul(A_sub, A_sub.t())
            D_sub = torch.matmul(concepts_t, A_sub.t())
            feature_frequencies_sub = A_sub.sum(dim=1, keepdim=True)
            feature_frequencies = feature_frequencies + feature_frequencies_sub
            feature_sums_sub = D_sub.sum(dim=1, keepdim=True)
            feature_sums = feature_sums + feature_sums_sub
            C = C + C_sub
            D = D + D_sub
            # by chatgpt:

            top_vals_in_batch, top_indices_in_batch = torch.topk(concepts_t, top_k, dim=1)
            gathered = concepts[top_indices_in_batch.reshape(-1)]
            topk_acts_in_batch = gathered.view(concepts_t.shape[0], top_k, concepts_t.shape[0])  # [C, K, C]
            # concepts_b: [B, C] — values for each token
            # For top-k per column:
            combined_vals = torch.cat([topk_values, top_vals_in_batch], dim=1)
            combined_acts = torch.cat([topk_patch_activations, topk_acts_in_batch], dim=1)  # [C, K + B, C]

            # Get top-k values and indices
            top_vals, top_indices = torch.topk(combined_vals, top_k, dim=1)  # [C, K]

            # Gather top-k activations
            topk_patch_activations = torch.gather(combined_acts, dim=1, index=top_indices.unsqueeze(-1).expand(-1, -1, num_concepts))  # [C, K, C]

            # Update state
            topk_values = top_vals
            
    C_normalized = C / feature_frequencies.repeat(1, num_concepts)

    # saving the matrices
    C_path = os.path.join(dir_path, "C.pth")
    C_normalized_path = os.path.join(dir_path, "C_normalized.pth")
    frequencies_path = os.path.join(dir_path, "feature_frequencies.pth")
    topk_patch_activations_path = os.path.join(dir_path, f"top{top_k}_patch_activations.pth")
    D_path = os.path.join(dir_path, "D.pth")
    sums_path = os.path.join(dir_path, "feature_sums.pth")
    torch.save(feature_frequencies, frequencies_path)
    
    torch.save(C, C_path)
    torch.save(C_normalized, C_normalized_path)
    torch.save(topk_patch_activations, topk_patch_activations_path)
    torch.save(feature_sums, sums_path)
    
    torch.save(D, D_path)

    print(f"Saved matrices at: {dir_path}")

if __name__ == "__main__":
    parser = arg_parser.get_default_parser()
    args = parser.parse_args()
    common_init(args)
    dir_path = os.path.join(args.probe_cs_save_dir, args.probe_split)
    activation_threshold = 0.001
    sub_dir_name = f"Athres_{activation_threshold}"
    sub_dir_path=  os.path.join(dir_path, sub_dir_name)
    try:
        os.makedirs(sub_dir_path, exist_ok=True)
    except OSError as error:
        print(f"Directory creation error: {error}")
    save_concept_strengths(args,sub_dir_path)

