import torch
import numpy as np
import os.path as osp
from cfm.utils import common_init, get_img_model
from dictionary_learning.utils import load_dictionary
import os
import clip
from tqdm import tqdm


class MethodBase:

    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        common_init(args, disable_make_dirs=True)

    def get_concepts(self):
        """ 
        Get all concept strengths in dataset split
        """
        raise NotImplementedError

    def get_logits(self):
        """ 
        Get all logits in dataset split
        """
        raise NotImplementedError

    def get_labels(self):
        """ 
        Get all labels in dataset split
        """
        if self.args.probe_split == "train":
            all_labels = torch.load(
                f"{self.args.probe_labels_dir['img']}/all_labels_train.pth")
            all_labels_train_val = torch.load(
                f"{self.args.probe_labels_dir['img']}/all_labels_train_val.pth")
            all_labels = torch.cat([all_labels, all_labels_train_val], dim=0)
        else:
            all_labels = torch.load(
                f"{self.args.probe_labels_dir['img']}/all_labels_{self.args.probe_split}.pth")
        return all_labels

    def get_concept_name(self, concept_idx):
        """ 
        Get concept text name given a concept ID
        """
        raise NotImplementedError

    def get_output_save_dir(self):
        """ 
        Get directory to save everything
        """
        raise NotImplementedError

    def get_classifier_weights(self):
        """ 
        Get linear classifier weights
        """
        raise NotImplementedError

    def get_concept_text_embedding(self, concept_idx):
        """ 
        Get text embedding given a concept ID
        """
        raise NotImplementedError

    def get_top_concept_indices_for_class(self, class_idx, num_indices=5):
        """
        Get top concept indices for each class based on coverage
        """
        raise NotImplementedError

    def get_concepts_from_features(self, x):
        """
        Given feature x, get concept
        """
        raise NotImplementedError

    def get_similarities(self):
        raise NotImplementedError

    def get_name_similarity(self, concept_idx):
        raise NotImplementedError

    def _get_contribs(self, split):
        suffix = f"_{split}.pt" if split == "train" else ".pt"
        all_concepts_fname = os.path.join(
            self.save_dir, f"all_concepts{suffix}")
        print(f"Loading {split} concepts from: {all_concepts_fname}")
        concepts = torch.load(all_concepts_fname)
        probe_labels_dir = self.args.probe_labels_dir["img"]
        if split == "train":
            labels_train = torch.load(os.path.join(
                probe_labels_dir, f"all_labels_train.pth"))
            labels_train_val = torch.load(
                osp.join(probe_labels_dir, f"all_labels_train_val.pth"))
            labels = torch.cat([labels_train, labels_train_val], dim=0)
        else:
            labels = torch.load(
                os.path.join(probe_labels_dir, f"all_labels_{split}.pth"))
        assert (len(concepts) == len(labels))
        print(f"Loading {split} labels from: {probe_labels_dir}")
        cum_sum_indices = torch.bincount(labels.long())
        sorting_order = labels.argsort()
        concepts = concepts[sorting_order]
        concepts_classwise = torch.split(
            concepts, cum_sum_indices.tolist())
        contribs_classwise = []
        for cidx, concept_classwise in enumerate(concepts_classwise):
            assert (concept_classwise.shape[1] ==
                    self.linear_layer_weights.shape[1])
            # For each class, multiply the concept strengths of images in that class with
            # weights of that class
            contribs_classwise.append(
                concept_classwise * self.linear_layer_weights[cidx].cpu())

        all_contribs = torch.stack(
            [c.clamp(min=0).sum(dim=0) for c in contribs_classwise])
        return all_contribs
    
class MethodCFM(MethodBase):

    def __init__(self, args, vocab_txt_path=None, embeddings_path=None, use_sae_from_args=True, **kwargs):
        self.args = args
        sae_config_to_use = self.args.config_name
        super().__init__(args, **kwargs)

        print(f"SAE config_used: {sae_config_to_use}")

        self.sae_load_dir = osp.join(
            args.save_dir["img"], "..", sae_config_to_use)
        self.load_dir = args.probe_cs_save_dir

        state_dict_path = os.path.join(
            self.sae_load_dir, "sae_checkpoints", 'sparse_autoencoder_final.pt')
        if self.args.sae_type== None: #not(self.args.use_topk):
            self.state_dict = torch.load(state_dict_path, map_location=args.device)

        self.concept_layer = self._get_concept_layer()
        if self.args.sae_type== "matryoshka":
            self.all_dic_vec= self.concept_layer.W_dec.detach().cpu().squeeze()
        else:
            raise NotImplementedError
        
        if embeddings_path is not None:
            assert vocab_txt_path is not None
            if type(embeddings_path) == str:
                self.all_embeddings = [torch.load(
                    embeddings_path, map_location='cpu')]
                self.vocab_txt_all = [np.genfromtxt(
                    vocab_txt_path, dtype=str, delimiter='\n')]
            elif type(embeddings_path) == list:
                self.all_embeddings = [torch.load(
                    e, map_location='cpu') for e in embeddings_path]
                self.vocab_txt_all = [np.genfromtxt(
                    t, dtype=str, delimiter='\n') for t in vocab_txt_path]
        else:
            if self.vocab_txt_path is not None:
                self.vocab_txt_all = [np.genfromtxt(
                    self.vocab_txt_path, dtype=str, delimiter='\n')]

    def get_concepts(self, split=None):
        if split is None:
            split = self.args.probe_split
        if split == "train":
            concept_path1 = os.path.join(
                self.load_dir, "train", "all_concepts.pth")
            concept_path2 = os.path.join(
                self.load_dir, "train_val", "all_concepts.pth")
            train_concepts = torch.load(concept_path1)
            train_val_concepts = torch.load(concept_path2)
            img_concepts = torch.cat(
                [train_concepts, train_val_concepts], dim=0)
        elif split== "only_train":
            concept_path = os.path.join(
                self.load_dir, "train", "all_concepts.pth")
            img_concepts = torch.load(concept_path)
        else:
            concept_path = os.path.join(
                self.load_dir, split, "all_concepts.pth")
            img_concepts = torch.load(concept_path)
        img_concepts = img_concepts.squeeze(1)
        return img_concepts

    def get_logits(self):
        assert self.args.probe_split == "val"
        all_logits = torch.load(os.path.join(
            self.load_dir, self.args.probe_config_name, "stats", self.args.probe_split, "val_all_logits.pt"), map_location=self.args.device)
        return all_logits

    def get_concept_name_similarity_matrix(self):
        all_similarities = []
        for vocab_idx, vocab_specific_embedding in enumerate(self.all_embeddings):
            vocab_specific_embedding = vocab_specific_embedding.to(
                torch.float32)
            dic_vec = self.all_dic_vec  # (n_features, n_concepts)
            dic_vec /= dic_vec.norm(dim=0, keepdim=True)
            similarities = torch.matmul(
                vocab_specific_embedding, dic_vec)
            all_similarities.append(similarities.detach().cpu())
        return all_similarities

    def get_concept_name(self, concept_idx=None, dic_vec=None, return_sim=False, return_vocab_id=False):
        output = []
        sim = []
        vocab_id = []

        for vocab_idx, vocab_specific_embedding in enumerate(self.all_embeddings):
            vocab_specific_embedding = vocab_specific_embedding.to(
                torch.float32)
            if concept_idx is not None:
                dic_vec = self.all_dic_vec[:, concept_idx].unsqueeze(1)
            elif dic_vec is not None:
                dic_vec = dic_vec.unsqueeze(1)
            dic_vec /= dic_vec.norm(dim=0, keepdim=True)
            similarities = torch.matmul(
                vocab_specific_embedding, dic_vec).squeeze(dim=1)
            top_index = torch.argmax(similarities)
            output.append(self.vocab_txt_all[vocab_idx][top_index])
            sim.append(similarities[top_index])
            vocab_id.append(top_index)

        if return_sim:
            if return_vocab_id:
                return output, sim, vocab_id
            else:
                return output, sim
        else:
            return output

    def get_output_save_dir(self):
        return os.path.join(
            self.load_dir, self.probe_config_dict[self.args.probe_dataset])

    def get_classifier_weights(self, probe_dataset=None, which_ckpt=None, checkpoint_save_path=None):
        num_concepts = self.args.autoencoder_input_dim_dict[self.args.ae_input_dim_dict_key["img"]
                                                            ] * self.args.expansion_factor
        # 
        if probe_dataset is None:
            probe_dataset = self.args.probe_dataset
        probe_dataset_nclasses_dict = {"places365": 365, 'imagenet': 1000, "coco_stuff": 171, 
                                "cc12m":1, "cityscapes": 19}
        num_classes = probe_dataset_nclasses_dict.get(probe_dataset, 0)
        
        # Use provided which_ckpt or default to self.args.which_ckpt
        # ckpt_name = which_ckpt if which_ckpt is not None else self.args.which_ckpt
        ckpt_name = which_ckpt 
        # Build the correct directory path
        # self.load_dir is typically: .../config_name/current_probe_dataset/
        # If probe_dataset is provided, replace the last part with the new dataset
        if checkpoint_save_path is not None:
            state_dict = torch.load(checkpoint_save_path, map_location=self.args.device)
        else:
            load_dir = self.load_dir
            if probe_dataset is not None:
                # Replace the last directory (current probe_dataset) with the new one
                load_dir = os.path.join(os.path.dirname(self.load_dir), probe_dataset)
            
            # Directory and file format: {probe_config_name}_{ckpt_name}
            # Example: lr0.0001_bs512_epo50_clCE_spL1_spl0.0_max_no_threshold
            config_with_ckpt = f"{self.args.probe_config_name}_{ckpt_name}"
            
            checkpoint_save_path = os.path.join(load_dir, config_with_ckpt, "on_concepts_ckpts")
            print(f"Loading classifier checkpoint from: {checkpoint_save_path}")
            
            checkpoint_file = f"on_concepts_final_{config_with_ckpt}.pt"
            print(f"Loading checkpoint: {checkpoint_file}")
            
            state_dict = torch.load(os.path.join(checkpoint_save_path, checkpoint_file), map_location=self.args.device)
        classifier = torch.nn.Linear(
            num_concepts, num_classes, bias=False).to(self.args.device)
        classifier.load_state_dict(state_dict['model'])
        return classifier.weight

    def get_concept_text_embedding(self, concept_idx, use_dic_vec=False):
        if use_dic_vec:
            return self.all_dic_vec[:, concept_idx]
        output = []
        for vocab_idx in range(len(self.all_embeddings)):
            output.append(self.all_selected_embeddings[vocab_idx][concept_idx])
        return output

    def get_similarities(self):
        return self.name_similarities

    def get_name_similarity(self, concept_idx):
        output = []
        for vocab_idx in range(len(self.all_embeddings)):
            output.append(self.name_similarities[vocab_idx][concept_idx])
        return output

    def get_top_concept_indices_for_class(self, class_idx, num_indices=5):
        if not hasattr(self, 'top_indices_for_all_classes'):
            self.top_indices_for_all_classes = self.state_dict[
                "global_stats"][self.args.mod_type]["cov"]["node_idxs"][2]
        if num_indices > 10:
            raise ValueError(
                "Can currently only provide up to top 10 concept indices for a class")
        return self.top_indices_for_all_classes[class_idx][:num_indices]

    def get_concepts_from_features(self, x):
        concepts, _ = self.concept_layer.forward(x)
        concepts = concepts.squeeze(1)
        return concepts

    def _decode_config(self, sae_config, probe_config):
        if sae_config is not None:
            sae_config = sae_config.split("_")
            for item in sae_config:
                if item.startswith("lr"):
                    self.args.lr = float(item[2:])
                elif item.startswith("l1coeff"):
                    self.args.l1_coeff = float(item[7:])
                elif item.startswith("ef"):
                    self.args.expansion_factor = int(item[2:])
                elif item.startswith("rf"):
                    self.args.resample_freq = int(item[2:])
                elif item.startswith("hook"):
                    self.args.hook_points = [str(item[4:])]
                elif item.startswith("bs"):
                    self.args.train_sae_bs = int(item[2:])
                elif item.startswith("epo"):
                    self.args.num_epochs = int(item[3:])
                else:
                    raise ValueError(f"Invalid SAE config item: {item}")
        if probe_config is not None:
            probe_config = probe_config.split("_")
            for item in probe_config:
                if item.startswith("lr"):
                    self.args.probe_lr = float(item[2:])
                elif item.startswith("bs"):
                    self.args.probe_train_bs = int(item[2:])
                elif item.startswith("epo"):
                    self.args.probe_epochs = int(item[3:])
                elif item.startswith("nobias"):
                    self.args.probe_bias = False
                elif item.startswith("cl"):
                    self.args.probe_classification_loss = str(item[2:])
                elif item.startswith("spl"):
                    self.args.probe_sparsity_loss_lambda = float(item[3:])
                elif item.startswith("sp"):
                    self.args.probe_sparsity_loss = str(item[2:])
                else:
                    raise ValueError(f"Invalid probe config item: {item}")
                
    def compute_concept_embeddings(self, method = "family weighted mean top patches mean norm child enforced", add_bias=False, applied_c_threshold = 0.8, activation_threshold = 0.001, matrix = "D", top_30 = False, autoencoder = None):
        concept_embedding = self.all_dic_vec
        if method != "vanilla":
            dir_path = os.path.join(self.args.probe_cs_save_dir, "train")
            sub_dir_name = f"Athres_{activation_threshold}"
            cooc_dir=  os.path.join(dir_path, sub_dir_name)
            if matrix == "D":
                C_norm = torch.load(os.path.join(cooc_dir, "D.pth"))
                C_norm = C_norm / C_norm.diag().unsqueeze(1).repeat(1, C_norm.shape[0])
            elif matrix == "C":
                C_norm = torch.load(os.path.join(cooc_dir, "C_normalized.pth"))
            C_thres = C_norm * (C_norm > applied_c_threshold)
            C_thres = C_thres.nan_to_num(0)
            C_thres.fill_diagonal_(1)
            if method == "family weighted mean top patches mean norm":
                if top_30:
                    topk_patch_activations = torch.load(os.path.join(cooc_dir, "top30_patch_activations.pth"))
                    print("using top 30 patches")
                else:
                    topk_patch_activations = torch.load(os.path.join(cooc_dir, "topk_patch_activations.pth"))
                top_patch_activations = topk_patch_activations.mean(axis=1)
                relevant_vectors = (C_thres > 0).float()
                vector_weighting = relevant_vectors.to(self.args.device) * top_patch_activations.to(self.args.device) # .T
                vector_weighting = vector_weighting/ vector_weighting.norm(dim=1, keepdim=True)
                vector_weighting = vector_weighting* top_patch_activations.norm(dim=1).unsqueeze(1).to(vector_weighting.device).mean()
            elif method == "family weighted mean top patches mean norm child enforced":
                if top_30:
                    topk_patch_activations = torch.load(os.path.join(cooc_dir, "top30_patch_activations.pth"))
                    print("using top 30 patches")
                else:
                    topk_patch_activations = torch.load(os.path.join(cooc_dir, "topk_patch_activations.pth"))
                top_patch_activations = topk_patch_activations.mean(axis=1)
                relevant_vectors = C_thres
                vector_weighting = relevant_vectors.to(self.args.device) * top_patch_activations.to(self.args.device) # .T
                vector_weighting = vector_weighting/ vector_weighting.norm(dim=1, keepdim=True)
                vector_weighting = vector_weighting* top_patch_activations.norm(dim=1).unsqueeze(1).to(vector_weighting.device).mean() * 1.33
                print("multiplied by 1.33 to lower bias children")
            elif method == "vanilla mean top patches mean norm":
                if top_30:
                    topk_patch_activations = torch.load(os.path.join(cooc_dir, "top30_patch_activations.pth"))
                    print("using top 30 patches")
                else:
                    topk_patch_activations = torch.load(os.path.join(cooc_dir, "topk_patch_activations.pth"))
                top_patch_activations = topk_patch_activations.mean(axis=1)
                relevant_vectors = torch.eye(C_thres.shape[0]).to(C_thres.device) 
                vector_weighting = relevant_vectors.to(self.args.device) * top_patch_activations.to(self.args.device) # .T
                vector_weighting = vector_weighting/ vector_weighting.norm(dim=1, keepdim=True)
                vector_weighting = vector_weighting* top_patch_activations.norm(dim=1).unsqueeze(1).to(vector_weighting.device).mean() * 1.33
            else:
                raise NotImplementedError
            
            # manual
            if add_bias:
                #vector_weighting
                if self.concept_layer is None:
                    autoencoder = self._get_concept_layer().to(self.args.device)
                else:
                    autoencoder = self.concept_layer.to(self.args.device)
                autoencoder.eval()
                concept_embedding = autoencoder.decode(vector_weighting.to(self.args.device)).detach()
            else:
                concept_embedding = torch.matmul(vector_weighting.to(self.args.device), concept_embedding.to(self.args.device))
        if add_bias and method== "vanilla":
            if self.concept_layer is None:
                autoencoder = self._get_concept_layer()
            else:
                autoencoder = self.concept_layer
            autoencoder.eval()
            concept_embedding = concept_embedding + autoencoder.b_dec.to(concept_embedding.device).detach()
        concept_embedding/= concept_embedding.norm(dim=1, keepdim=True)
        return concept_embedding

    def _get_concept_layer(self):
        if self.args.sae_type!= None:
            base_path = os.path.join(self.args.save_dir_sae_ckpts["img"], self.args.save_suffix) +  self.args.config_name+ "/" + "trainer_0"
            print("loading autoencoder from " + base_path)
            autoencoder, config =  load_dictionary(base_path, self.args.device)
        else:     
            raise NotImplementedError
        return autoencoder
    def create_multi_template_embeddings(self, vocab_texts, vocab_name, model, batch_size=1024, force_recompute=False):
        """
        Create embeddings using noun, adjective, and verb templates.
        Loads from cache if available, otherwise computes and saves.
        
        Args:
            vocab_texts: Array of vocabulary texts
            vocab_name: Name identifier for this vocabulary (e.g., "clipdissect_20k")
            model: Model with clip_backbone.backbone.encode_text
            batch_size: Batch size for encoding
            force_recompute: If True, recompute even if cache exists
        
        Returns:
            noun_embeds, adj_embeds, verb_embeds: Tensors of shape [num_vocab, embed_dim]
        """
        if self.args.img_enc_name_for_saving== "dinoclip_ViT-B16":
            clip_name = "clip_ViT-B16"
        elif self.args.img_enc_name_for_saving== "dinoclip_openai_ViT-L14@336":
            clip_name = "dinoclip_openai_ViT-L14@336"
        else:
            clip_name = self.args.img_enc_name_for_saving
        # Define cache paths
        noun_path = osp.join(self.args.vocab_dir, f"{vocab_name}_{clip_name}_noun_embeds.pt")
        adj_path = osp.join(self.args.vocab_dir, f"{vocab_name}_{clip_name}_adj_embeds.pt")
        verb_path = osp.join(self.args.vocab_dir, f"{vocab_name}_{clip_name}_verb_embeds.pt")

        # Check if all embeddings exist
        if not force_recompute and osp.exists(noun_path) and osp.exists(adj_path) and osp.exists(verb_path):
            print(f"Loading cached template embeddings for {vocab_name}...")
            noun_embeds = torch.load(noun_path, map_location='cpu')
            adj_embeds = torch.load(adj_path, map_location='cpu')
            verb_embeds = torch.load(verb_path, map_location='cpu')
            return noun_embeds, adj_embeds, verb_embeds
        
        # Otherwise compute embeddings
        print(f"Computing template embeddings for {vocab_name}...")
        
        noun_templates = [
            "{}",
            "a {}",
            "the {}",
            "one {}",
            "a photo of a {}",
            "a picture of a {}",
            "a close view of a {}",
            "an example of a {}",
            "something called {}",
            "a type of {}",
        ]

        adjective_templates = [
            "{}",
            "something that is {}",
            "object that is {}",
            "appearance that is {}",
            "person that is {}",
            "something that looks {}",
            "someone that looks {}",
            "a {} object",
            "a {} thing",
            "a {} person",
        ]

        verb_templates = [
            "{}",
            "to {}",
            "someone who is {}",
            "something that is {}",
            "a person {}",
            "a thing {}",
            "a photo of someone {}",
            "a photo of something {}",
            "a picture of someone {}",
            "a picture of something {}",
        ]
        
        def encode_with_templates(vocab_texts, templates):
            """Helper to encode vocab with specific templates"""
            num_vocab = len(vocab_texts)
            num_templates = len(templates)
            total_texts = num_vocab * num_templates
            
            # Create all templated texts
            all_templated_texts = [
                template.format(vocab_text)
                for vocab_text in vocab_texts
                for template in templates
            ]
            
            # Tokenize
            text_inputs = torch.cat([clip.tokenize(text) for text in all_templated_texts])
            
            # Get embedding dimension
            with torch.no_grad():
                sample_embed = model.clip_backbone.backbone.encode_text(text_inputs[:1].to(self.args.device))
                embed_dim = sample_embed.shape[1]
            
            # Pre-allocate output tensor
            all_embeds = torch.zeros(total_texts, embed_dim)
            
            # Encode in batches
            for i in tqdm(range(0, total_texts, batch_size), desc="Encoding"):
                end_idx = min(i + batch_size, total_texts)
                text_batch = text_inputs[i:end_idx].to(self.args.device)
                
                with torch.no_grad():
                    batch_embeds = model.clip_backbone.backbone.encode_text(text_batch)
                    batch_embeds = batch_embeds / batch_embeds.norm(dim=-1, keepdim=True)
                    all_embeds[i:end_idx] = batch_embeds.cpu()
            
            # Reshape and average across templates
            all_embeds = all_embeds.view(num_vocab, num_templates, embed_dim)
            averaged_embeds = all_embeds.mean(dim=1)
            averaged_embeds = averaged_embeds / averaged_embeds.norm(dim=-1, keepdim=True)
            
            return averaged_embeds
        
        noun_embeds = encode_with_templates(vocab_texts, noun_templates)
        adj_embeds = encode_with_templates(vocab_texts, adjective_templates)
        verb_embeds = encode_with_templates(vocab_texts, verb_templates)
        
        # Save embeddings
        print(f"Saving template embeddings for {vocab_name}...")
        torch.save(noun_embeds, noun_path)
        torch.save(adj_embeds, adj_path)
        torch.save(verb_embeds, verb_path)
        
        return noun_embeds, adj_embeds, verb_embeds
    
    def _compute_all_concept_embeddings_texts_and_indices(self, method="family weighted", add_bias=False, applied_c_threshold=0.8, activation_threshold=0.001, matrix="D", top_30=False):
        multi_vocab_all_selected_embeddings = []
        multi_vocab_all_selected_indices = []
        multi_vocab_all_texts = []
        multi_vocab_all_name_similarities = []
        multi_vocab_all_concept_types = []
        multi_vocab_topk_selected_embeddings = []
        multi_vocab_topk_selected_indices = []
        multi_vocab_topk_texts = []
        multi_vocab_topk_name_similarities = []
        multi_vocab_topk_concept_types = []
        k = 6
        
        model, _ = get_img_model(self.args)
        model = model.to(self.args.device)
        model.eval()
        
        # Extract vocab names from paths
        vocab_names = []
        vocab_txt_paths = [
            osp.join(self.args.vocab_dir, "clipdissect_20k.txt"),
            osp.join(self.args.vocab_dir, "concreteness_40k.txt"),
            osp.join(self.args.vocab_dir, "concreteness_60k.txt"),
            osp.join(self.args.vocab_dir, "laion_uniwords4000.txt"),
            osp.join(self.args.vocab_dir, "laion_bigrams.txt")
        ]
        
        for path in vocab_txt_paths:
            vocab_name = osp.basename(path).replace('.txt', '')
            vocab_names.append(vocab_name)
        
        # Create template embeddings for all vocabularies
        print("Creating/loading template embeddings for all vocabularies...")
        all_noun_embeds = []
        all_adj_embeds = []
        all_verb_embeds = []
        
        for vocab_idx, vocab_texts in enumerate(self.vocab_txt_all):
            noun_embeds, adj_embeds, verb_embeds = self.create_multi_template_embeddings(
                vocab_texts,
                vocab_names[vocab_idx],
                model,
                batch_size=1024,
                force_recompute=False
            )
            all_noun_embeds.append(noun_embeds)
            all_adj_embeds.append(adj_embeds)
            all_verb_embeds.append(verb_embeds)
        
        # Rest of your existing code for computing concept_embedding...
        concept_embedding = self.compute_concept_embeddings(method = method, add_bias=add_bias, applied_c_threshold = applied_c_threshold, activation_threshold = activation_threshold, matrix = matrix, top_30 = top_30)
        # Now use the template embeddings for matching
        for vocab_idx in range(len(self.vocab_txt_all)):
            vocab_texts = self.vocab_txt_all[vocab_idx]
            noun_embeds = all_noun_embeds[vocab_idx]
            adj_embeds = all_adj_embeds[vocab_idx]
            verb_embeds = all_verb_embeds[vocab_idx]
            
            # Compute similarities for each template type
            concept_embedding_device = concept_embedding.to(self.args.device)
            
            noun_similarities = torch.matmul(
                noun_embeds.to(self.args.device),
                concept_embedding_device.T
            ).cpu()
            
            adj_similarities = torch.matmul(
                adj_embeds.to(self.args.device),
                concept_embedding_device.T
            ).cpu()
            
            verb_similarities = torch.matmul(
                verb_embeds.to(self.args.device),
                concept_embedding_device.T
            ).cpu()
            
            # Stack all similarities: [3, num_vocab, num_concepts]
            all_similarities = torch.stack([noun_similarities, adj_similarities, verb_similarities], dim=0)
            
            # Find max across vocab items for each type: [3, num_concepts]
            max_sim_per_type, max_idx_per_type = all_similarities.max(dim=1)
            
            # Find which type gives maximum similarity for each concept: [num_concepts]
            best_type_idx = max_sim_per_type.argmax(dim=0)
            
            # Get the corresponding vocab indices and similarities
            concept_indices = torch.arange(all_similarities.shape[2])
            all_selected_indices = max_idx_per_type[best_type_idx, concept_indices]
            all_name_similarities = max_sim_per_type[best_type_idx, concept_indices]
            
            # Get the actual embeddings and texts
            type_embeds_list = [noun_embeds, adj_embeds, verb_embeds]
            all_selected_embeddings = torch.stack([
                type_embeds_list[type_idx][vocab_idx]
                for type_idx, vocab_idx in zip(best_type_idx, all_selected_indices)
            ])
            
            all_texts = vocab_texts[all_selected_indices.numpy()]
            all_concept_types = best_type_idx
            
            # Top-k: for each concept, get top-k across ALL template types and vocab
            topk_values_list = []
            topk_indices_list = []
            topk_type_list = []
            
            for concept_idx in range(all_similarities.shape[2]):
                concept_sims = all_similarities[:, :, concept_idx]
                flat_sims = concept_sims.flatten()
                topk_vals, topk_flat_idx = flat_sims.topk(k)
                
                topk_type_idx = topk_flat_idx // len(vocab_texts)
                topk_vocab_idx = topk_flat_idx % len(vocab_texts)
                
                topk_values_list.append(topk_vals)
                topk_indices_list.append(topk_vocab_idx)
                topk_type_list.append(topk_type_idx)
            
            topk_name_similarities = torch.stack(topk_values_list)
            topk_selected_indices = torch.stack(topk_indices_list)
            topk_concept_types = torch.stack(topk_type_list)
            
            # Get embeddings and texts for top-k
            topk_selected_embeddings = torch.stack([
                torch.stack([type_embeds_list[type_idx][vocab_idx] 
                            for type_idx, vocab_idx in zip(topk_concept_types[i], topk_selected_indices[i])])
                for i in range(len(topk_selected_indices))
            ])

            topk_texts = [
                vocab_texts[topk_selected_indices[i].numpy()]
                for i in range(len(topk_selected_indices))
            ]
            topk_texts = np.array(topk_texts)
            
            # Append to multi-vocab lists
            multi_vocab_all_selected_embeddings.append(all_selected_embeddings)
            multi_vocab_all_selected_indices.append(all_selected_indices)
            multi_vocab_all_texts.append(all_texts)
            multi_vocab_all_name_similarities.append(all_name_similarities)
            multi_vocab_all_concept_types.append(all_concept_types)
            
            multi_vocab_topk_selected_embeddings.append(topk_selected_embeddings)
            multi_vocab_topk_selected_indices.append(topk_selected_indices)
            multi_vocab_topk_texts.append(topk_texts)
            multi_vocab_topk_name_similarities.append(topk_name_similarities)
            multi_vocab_topk_concept_types.append(topk_concept_types)
            
            # Print statistics
            type_names = ['noun', 'adjective', 'verb']
            type_counts = [(best_type_idx == i).sum().item() for i in range(3)]
            print(f"Vocab {vocab_idx} - Concept type distribution: {dict(zip(type_names, type_counts))}")
        
        self.topk_selected_embeddings = multi_vocab_topk_selected_embeddings
        self.topk_selected_indices = multi_vocab_topk_selected_indices
        self.topk_vocab_txt_selected = multi_vocab_topk_texts    # topk concepts per vocab
        self.topk_name_similarities = multi_vocab_topk_name_similarities
        self.topk_concept_types = multi_vocab_topk_concept_types  
        # to assign 1 name
        assigned_concept_names = []
        for i in range(len(self.topk_vocab_txt_selected[0])):
            best_sim = 0
            best_name = ""
            for j in range(len(self.topk_vocab_txt_selected)):
                for l in range(len(self.topk_vocab_txt_selected[j][i])):
                    if self.topk_name_similarities[j][i][l] > best_sim:
                        best_sim = self.topk_name_similarities[j][i][l]
                        best_name = self.topk_vocab_txt_selected[j][i][l]
            assigned_concept_names.append(str(best_name))
        self.assigned_concept_names = assigned_concept_names # final concept names
    

def get_method(method, args, **kwargs):
    if method == "cfm":
        return MethodCFM(args, **kwargs)
    else:
        raise ValueError(f"Invalid method: {method}")
