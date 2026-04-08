# from dncbm.custom_pipeline import Pipeline
import os
import torch
import numpy as np
# import math
import psutil

from time import time
from torch.utils.data import Dataset, DataLoader
import json
import pickle
import multiprocessing as mp
# cfm
from cfm import arg_parser
from cfm.utils import common_init
from cfm.data_utils.activation_store.tensor_store import TensorActivationStore
from cfm.sae_training import trainSAE
# dictionary_learning 
from dictionary_learning.trainers.top_k import TopKTrainer, AutoEncoderTopK
from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKTrainer, MatryoshkaBatchTopKSAE
from dictionary_learning.trainers.batch_top_k import BatchTopKTrainer, BatchTopKSAE




def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / 1024 ** 3:.2f} GB")

parser = arg_parser.get_default_parser()
args = parser.parse_args()

print(f"Received hyperparameters:\n"
      f"  lr: {args.lr}\n"
      f"  expansion_factor: {args.expansion_factor}\n"
      f"  k: {args.k}\n"
      f"  matryoshka_fractions: {args.matryoshka_fractions}\n"
      f"  train_sae_bs: {args.train_sae_bs}\n"
      f"  img_enc_name: {args.img_enc_name}\n"
      )

common_init(args)

print("args: ", args)
start_time = time()

# parameters of sae
autoencoder_input_dim: int = args.autoencoder_input_dim_dict[
    args.ae_input_dim_dict_key["img"]]
hidden_layer_size = int(autoencoder_input_dim * args.expansion_factor)
# create the autoencoder

print(f"Autoencoder created at {time() - start_time} seconds")



print(
    f'------------Getting Image activations from directory: {args.data_dir_activations["img"]}')
print(f"------------Getting Image activations from model: {args.img_enc_name}")

fnames = os.listdir(args.data_dir_activations["img"]) # had to add .pth to the path
embedding_dir = args.data_dir_activations["img"]
print(f'Getting fnames from {args.data_dir_activations["img"]}')
if args.sae_dataset == "imagenet" or args.sae_dataset == "places365" or args.sae_dataset == "cc12m": # only large datasets are saved as memmaps
    is_memmap = True
else:
    is_memmap = False
if not(is_memmap):   
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

    train_loader = DataLoader(train_dataset, batch_size=args.train_sae_bs, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(trainval_dataset, batch_size=32000, shuffle=False, collate_fn=custom_collate_fn)
    # args.train_sae_bs*4
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
    class ActivationDataset(Dataset):
        def __init__(self, path, autoencoder_input_dim):
            self.memmap = load_memmap_with_metadata(path)
            # self.memmap = np.memmap(path, dtype=dtype, mode='r', shape=shape)
            #self.memmap = self.memmap.reshape(-1, autoencoder_input_dim)
            print(self.memmap.shape)
        def __len__(self):
            return self.memmap.shape[0] 
        
        def get_total_samples(self):
            return self.memmap.shape[0] * self.memmap.shape[1]

        def __getitem__(self, idx):
            item = self.memmap[idx]
            return torch.as_tensor(item.copy(), dtype=torch.float32)
    def collate_fn(batch):
        batch = torch.stack(batch, dim=0)
        return batch.reshape(-1, batch.shape[-1])


    # dataset
    train_split_path = os.path.join(args.data_dir_activations["img"], "train.dat")
    train_val_split_path = os.path.join(args.data_dir_activations["img"], "train_val.dat")

    print("Using ActivationDataset with memmap")
    train_dataset = ActivationDataset(train_split_path, autoencoder_input_dim)
    trainval_dataset = ActivationDataset(train_val_split_path, autoencoder_input_dim)
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
    print(f"SLURM_CPUS_PER_TASK: {slurm_cpus}")
    num_workers = min(slurm_cpus // 2, 8)
    print(f"num_workers: {num_workers}")
    print(f"total number of workers: {mp.cpu_count()}")

    train_loader = DataLoader(train_dataset, batch_size=args.train_sae_bs//196, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=6, collate_fn=collate_fn)
    val_loader = DataLoader(trainval_dataset, batch_size=(args.train_sae_bs*4)//196, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True, prefetch_factor=4, collate_fn=collate_fn)
    number_of_samples = train_dataset.get_total_samples()
    print("training samples:", number_of_samples)
    number_of_trainval_samples = trainval_dataset.get_total_samples()
    print("train val samples:", number_of_trainval_samples)

print(f"Dataset size: {number_of_samples}, Batch size: {args.train_sae_bs}")

# wandb
wandb_project_name = None
if args.use_wandb:
    # wandb_project_name = f"SAE_k_{args.k}_ea_{args.expansion_factor}_lr_{args.lr}_bs_{args.train_sae_bs}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}{args.save_suffix}"
    wandb_project_name = args.config_name # f"k_{args.k}_ef_{args.expansion_factor}_lr_{args.lr}_mf_{args.matryoshka_fractions}"
    print(f"wandb project name: {wandb_project_name}")

wandb_project = f"SAE_{args.sae_dataset}_{args.img_enc_name_for_saving}_{args.sae_type}"

MODEL_NAME = args.img_enc_name
RANDOM_SEED = 42
LAYER = 3  # should not have any influence but had to set
EVAL_TOLERANCE = 0.01

decay_start = None
submodule_name = "patch_tokens_of_last_layer"
steps =int(number_of_samples/((args.train_sae_bs//196)*196)) # 7936 # was the one for the normal sae maybe do int(num_tokens / sae_batch_size)
print(f"sae_type: {args.sae_type}")
print(f"steps: {steps}")
print(f"epochs: {args.num_epochs}")
print(f"number_of_samples: {number_of_samples}")
print(f"num_epochs: {args.num_epochs}")
if args.ckpt_freq ==0:
    save_steps = None
else:
    save_steps = args.ckpt_freq * number_of_samples
# resample_steps = args.resample_freq * number_of_samples
# l1 might be missing

# train sae
if args.sae_type == "topk":
    k = args.k
    auxk_alpha = 1 / 32
    trainer_cfg = {  # todo add the rest of the parameters
        "trainer": TopKTrainer,
        "dict_class": AutoEncoderTopK,
        "activation_dim": autoencoder_input_dim,
        "dict_size": hidden_layer_size,
        "k": k,
        "auxk_alpha": auxk_alpha,
        "warmup_steps": 0,
        "steps": steps,
        "decay_start": decay_start,
        "lr": float(args.lr),
        "device": args.device,
        "seed": RANDOM_SEED,
        "wandb_name": wandb_project_name,
        "layer": LAYER,
        "lm_name": MODEL_NAME,
        "submodule_name": submodule_name,
        # "resample_steps": resample_steps, I think this is not needed by topk
    }

    log_steps = 100
    print("starting training")
    ae = trainSAE(
        data=train_loader,  
        trainer_configs=[trainer_cfg],
        steps=steps,
        save_steps=save_steps,
        save_dir=os.path.join(args.save_dir_sae_ckpts["img"], wandb_project_name),
        use_wandb=args.use_wandb,
        log_steps=log_steps,
        # normalize_activations = True
        wandb_project = wandb_project,
        epochs=args.num_epochs
    )
elif args.sae_type == "matryoshka":
    k = args.k
    warmup_steps= 0# TODO change to 500
    
    # thresholdbeta and threshold_start can also be set, aswell as top_k_aux
    group_fractions = json.loads(args.matryoshka_fractions)   # [0.01, 0.1, 0.89]
    if args.sae_dataset == "cc12m":
        dead_feature_threshold = 50_000_000
    else:
        dead_feature_threshold = 10_000_000
    trainer_cfg = {
            "trainer": MatryoshkaBatchTopKTrainer,
            "dict_class": MatryoshkaBatchTopKSAE,
            "lr": float(args.lr),
            "steps": steps,
            "warmup_steps": warmup_steps,
            "decay_start": decay_start,
            "seed": RANDOM_SEED,
            "activation_dim": autoencoder_input_dim,
            "dict_size": hidden_layer_size,
            "group_fractions": group_fractions,
            # "group_weights": self.group_weights, # can be computed from group_fractions and dict size
            # "group_sizes": self.group_sizes,
            "k": k,
            "device": args.device,
            "layer": LAYER,
            "lm_name": MODEL_NAME,
            "wandb_name": wandb_project_name + "dft_" + str(dead_feature_threshold),
            "submodule_name": submodule_name,
            # "add_bias": not(args.no_bias),  # whether to add bias to the autoencoder
            # "dead_feature_threshold": dead_feature_threshold,  # steps so that its always one epoch
        }
    
    print("trainer_cfg: ", trainer_cfg)
    log_steps = 2000
    print("starting training")
    ae = trainSAE(
        data=train_loader,  
        trainer_configs=[trainer_cfg],
        steps=steps,
        save_steps=save_steps,
        save_dir=os.path.join(args.save_dir_sae_ckpts["img"], wandb_project_name),
        use_wandb=args.use_wandb,
        log_steps=log_steps,
        wandb_project = wandb_project,
        val_data= val_loader,
        epochs=args.num_epochs,
        dead_feature_threshold=dead_feature_threshold
        # normalize_activations = True
    )
elif args.sae_type == "batch_topk":
    k = args.k
    # thresholdbeta and threshold_start can also be set, aswell as top_k_aux
    trainer_cfg = {  # todo add the rest of the parameters
        "trainer": BatchTopKTrainer,
        "dict_class": BatchTopKSAE,
        "activation_dim": autoencoder_input_dim,
        "dict_size": hidden_layer_size,
        "k": k,
        "warmup_steps": 0,
        "steps": steps,
        "decay_start": decay_start,
        "lr": float(args.lr),
        "device": args.device,
        "seed": RANDOM_SEED,
        "wandb_name": wandb_project_name,
        "layer": LAYER,
        "lm_name": MODEL_NAME,
        "submodule_name": submodule_name,
    }
    log_steps = 100
    print("starting training")
    ae = trainSAE(
        data=train_loader,  
        trainer_configs=[trainer_cfg],
        steps=steps,
        save_steps=save_steps,
        save_dir=os.path.join(args.save_dir_sae_ckpts["img"], wandb_project_name),
        use_wandb=args.use_wandb,
        log_steps=log_steps,
        wandb_project = wandb_project,
        epochs=args.num_epochs
        # normalize_activations = True
    )
else: 
    print('Did not implement type of SAE')
    raise NotImplementedError

print(f"-------total time taken------ {np.round(time()-start_time,3)}")