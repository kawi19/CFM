import argparse

def get_default_parser():
    parser = argparse.ArgumentParser()
    # Adam parameters (set to the default ones here)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adam_beta_1", type=float, default=0.9)
    parser.add_argument("--adam_beta_2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)

    parser.add_argument("--img_enc_name", type=str, default='dinoclip_ViT-B/16',
                        help='Name of the clip image encoder', choices=["clip_ViT-B/16",  "dinoclip_ViT-B/16", "dinoclip_openai_ViT-B/16"])
    parser.add_argument('--hook_points', nargs='*',
                        help='Name of the model hook points to get the activations from', default=['out'])
    parser.add_argument("--val_freq", type=int, default=1,help='number of samples after which to run validation')
    parser.add_argument("--ckpt_freq", type=int, default=0,   # TODO maybe delete
                        help='number of samples after which to save the checkpoint')

    # SAE related
    parser.add_argument("--sae_dataset", type=str, default="cc12m")
    parser.add_argument("--train_sae_bs", type=int, default=16384,
                        help="batch size to train SAE")
    parser.add_argument("--expansion_factor", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="number of epochs to train the SAE")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--save_suffix", type=str, default="")
    parser.add_argument("--sae_type", type=str, default="matryoshka")
    parser.add_argument("--k", type=int, default=12,help='batch top-k parameter of topk sae')
    parser.add_argument("--matryoshka_fractions", type=str, default="[0.008,0.03,0.06,0.12,0.24,0.542]",help='proportions of matryoshka sections, need to add up to 1')
    parser.add_argument("--no_bias", action="store_true", default=False,help='whether to add bias to the autoencoder')
    parser.add_argument("--use_threshold", action="store_true", default=False, help='whether to use the learned activation threshold from the training of the SAE at inference (related to batch-top-k)')

    # linear probe related
    parser.add_argument("--probe_dataset", type=str, default="imagenet")
    parser.add_argument("--probe_save_dir", type=str,
                        default="")
    parser.add_argument("--probe_lr", type=float, default=1e-4)
    parser.add_argument("--probe_train_bs", type=int, default=512)
    parser.add_argument("--probe_epochs", type=int, default=100,
                        help="number of epochs to train the linear probe")
    parser.add_argument("--probe_nclasses", type=int, default=1000,
                        help="number of classes in the probe dataset")
    parser.add_argument("--probe_on_features",
                        action="store_true", default=False, help="train probe on features you get from the feature extractor")
    parser.add_argument("--probe_split", type=str, default="train",
                        help="which split of the probe dataset to use for training or for analysis depending on the context")
    parser.add_argument("--probe_classification_loss",
                        type=str, default="CE", choices=["CE", "BCE"])
    parser.add_argument("--probe_sparsity_loss", type=str,
                        default=None, choices=["L1"])
    parser.add_argument("--probe_sparsity_loss_lambda", type=float, default=0)
    parser.add_argument("--probe_val_freq", type=int, default=1)
    parser.add_argument("--probe_eval_coverage_freq", type=int, default=1)
    parser.add_argument("--probe_aggregation_method", type=str, default="max", choices=["max", "sum", "mean", "softmax_max", "mean_max", "mean_max_var", "attn_single", "attn_multi", "attn_query", "attn_query_max"])

    parser.add_argument("--use_wandb", action="store_true", default=False)    
    
    return parser
