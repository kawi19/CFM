from cfm import arg_parser
from cfm.utils import common_init
import os.path as osp
from cfm import method_utils

def save_concept_names(args):
    # get names
    if args.img_enc_name_for_saving== "dinoclip_ViT-B16":
        clip_name = "ViT-B16"
    else:
        clip_name = args.img_enc_name_for_saving
    embeddings_path = [osp.join(args.vocab_dir,f"embeddings_clip_{clip_name}_clipdissect_20k.pth" ), osp.join(args.vocab_dir,f"embeddings_clip_{clip_name}_concreteness_40k.pth"), osp.join(args.vocab_dir,f"embeddings_clip_{clip_name}_concreteness_60k.pth"), osp.join(args.vocab_dir,f"embeddings_clip_{clip_name}_laion_uniwords4000.pth"), osp.join(args.vocab_dir,f"embeddings_clip_{clip_name}_laion_bigrams.pth")] 
    vocab_txt_path  = [osp.join(args.vocab_dir, "clipdissect_20k.txt"), osp.join(args.vocab_dir, "concreteness_40k.txt"), osp.join(args.vocab_dir, "concreteness_60k_filtered_3words.txt"), osp.join(args.vocab_dir, "laion_uniwords4000.txt"), osp.join(args.vocab_dir, "laion_bigrams.txt")]
    method_obj = method_utils.get_method("cfm", args, embeddings_path=embeddings_path, vocab_txt_path=vocab_txt_path, use_fixed_sae=False)
    method_obj._compute_all_concept_embeddings_texts_and_indices(add_bias=True, method= "family weighted mean top patches mean norm child enforced", top_30=True, applied_c_threshold=0.75) 
    assigned_concept_names = method_obj.assigned_concept_names
    # save names
    save_path = osp.join(osp.join(args.save_dir_sae_ckpts['img'], args.save_suffix) + args.config_name + "/" + 'trainer_0' , "concept_names.txt")
    print(f"Saving concept names to {save_path}")
    with open(save_path, 'w') as f:
        for name in assigned_concept_names:
            f.write("%s\n" % name)


if __name__ == '__main__':
    # Run this file if you want to save the concept names of a SAE
    parser = arg_parser.get_default_parser()
    args = parser.parse_args()
    common_init(args)
    save_concept_names(args)