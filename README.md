# Insight: Interpretable Semantic Hierarchies in Vision-Language Encoders

<a href="https://explainablemachines.com/members/kai-wittenmayer.html">Kai Wittenmayer</a>,
<a href="https://sukrutrao.github.io">Sukrut Rao</a>,
<a href="https://m-parchami.github.io">Amin Parchami-Araghi</a>,
<a href="https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/bernt-schiele">Bernt Schiele</a>,
<a href="https://explainablemachines.com/members/jonas-fischer.html">Jonas Fischer</a>

Max Planck Institute for Informatics, Saarland Informatics Campus, Saarbrücken, Germany  

<p align="center">
  <a href="assets/teaser.png">
    <img src="assets/teaser.png" width="100%">
  </a>
</p>

---

[![arXiv](https://img.shields.io/badge/arXiv-2601.13798-b31b1b.svg)](https://arxiv.org/abs/2601.13798)


---

## 📰 News
* **[Apr 2026]**: Initial codebase released.
* **[Jan 2026]**: Our arXiv preprint is now available.
---

## Contents
- [Abstract](#abstract)
- [Code](#code)
- [Citation](#citation)

## Abstract

Language-aligned vision foundation models perform strongly across diverse downstream tasks. Yet, their learned representations remain opaque, making interpreting their decision-making difficult. 
Recent work decompose these representations into human-interpretable concepts, but provide poor spatial grounding and are limited to image classification tasks.
In this work, we propose CFM, a *language-aligned concept foundation model for vision* that provides fine-grained concepts, which are human-interpretable and spatially grounded in the input image.
When paired with a foundation model with strong semantic representations, we get explanations for *any of its downstream tasks*. Examining local co-occurrence dependencies of concepts allows us to define concept relationships through which we improve concept naming and obtain richer explanations.
On benchmark data, we show that CFM provides performance on classification, segmentation, and captioning that is competitive with opaque foundation models while providing fine-grained, high quality concept-based explanations.

---
## Set-up
### Prerequisites
- Miniforge/conda installed.
- Python 3.11 
- Recommended disk space: 4.5 TBs (For saving all dense features for SAE training) 

### Installing the Packages

Follow this workflow below to create a environment and
install the Python dependencies used by this project.

Conda / Miniforge:

```bash
# create the environment from the repository environment.yml
conda env create -f environment.yml

# activate
conda activate cfm-env

# install any remaining pip packages
pip install -r requirements.txt

# install cfm
pip install -e .
```
### CLIP-DINOiser Backbone
This repository requires the [CLIP-DINOiser](https://github.com/wysoczanska/clip_dinoiser) backbone. Download the `last.pt` checkpoint [here](https://github.com/wysoczanska/clip_dinoiser/blob/main/checkpoints/last.pt) and place it at `cfm/clip_dinoiser_backbone/checkpoints/last.pt`.
### Dataset for training Sparse Autoencoder (CC12M)

Download the Conceptual 12M (CC12M) dataset directly from Hugging Face: [conceptual_12m](https://huggingface.co/datasets/conceptual_12m).

Once downloaded, configure the dataset path in `config.py`.
### Vocabularies for naming concepts
We use the following vocabularies for naming concepts:
* From [CLIP-Dissect](https://arxiv.org/abs/2204.10965) download [20k.txt](https://github.com/first20hours/google-10000-english/blob/master/20k.txt) and place in the vocab_dir as "clipdissect_20k.txt"
Additional custom vocabularies will be released soon. 

Set the path to vocab_dir in `config.py`.
### Datasets for training downstream classification probes, open-vocab segmentation evaluation and image captioning 
#### Classification probes datasets
These are the datasets on which linear probes are trained on the learnt concept bottleneck to form a concept bottleneck model (CBM). In our paper, we use 2 datasets: Places365, ImageNet. Instructions for running experiments on these datasets are provided below, for other datasets you may need to define your own utils.

* Download the respective datasets:
    * [Places365](https://pytorch.org/vision/main/generated/torchvision.datasets.Places365.html)
    * [ImageNet](https://www.image-net.org/)
* Set the paths to the datasets in `config.py`.
#### Segmentation evaluation datasets
Please follow the [MMSeg data preparation document](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md). 
```shell
dataset
├── ADE
│   ├── ADEChallengeData2016
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
├── Cityscapes
│   ├── leftImg8bit
│   │   ├── train
│   │   ├── val
│   ├── gtFine
│   │   ├── train
│   │   ├── val
├── ms_coco_17
│   ├── images
│   │   ├── train2017
│   │   ├── val2017
│   ├── annotations
│   │   ├── object
│   │   ├── stuff
├── PascalVOC
│   ├── VOC2012
│   │   ├── JPEGImages
│   │   ├── SegmentationClass
│   │   ├── ImageSets
│   │   │   ├── Segmentation
│   ├── VOC2010
│   │   ├── JPEGImages
│   │   ├── SegmentationClassContext
│   │   ├── ImageSets
│   │   │   ├── SegmentationContext
│   │   │   │   ├── train.txt
│   │   │   │   ├── val.txt
```
#### Image Captioning datasets
We use MS-COCO for evaluating the captioning performance of our concept bottleneck model.
* Download the images and annotations:
    * Download [train_captions](https://drive.google.com/file/d/1D3EzUK1d1lNhD2hAvRiKPThidiVbP2K_/view?usp=sharing) to `data/coco/annotations`.
    * Download [training images](http://images.cocodataset.org/zips/train2014.zip) and [validation images](http://images.cocodataset.org/zips/val2014.zip) and unzip (We use Karpathy et el. split).
* Set the paths to the datasets in `config.py`.
## Usage
### Training (Can be skipped if you want to directly use the provided checkpoints and assigned names for concepts)
#### Concept Mapping: Training a Matryoshka BatchTopK Sparse Autoencoder (SAE)
##### Save the dense CLIP features on CC12M to train the SAE on
```
python scripts/save_dense_features.py --img_enc_name dinoclip_ViT-B/16 --probe_dataset cc12m
```
##### Train the SAE
```
python scripts/train_dict_sae.py --img_enc_name dinoclip_ViT-B/16 --use_wandb
```
#### Concept Relation Discovery
```
python scripts/save_conf_matrix_and_top_act.py --probe_dataset cc12m --probe_split "train" 
```
#### Concept Naming
```
python scripts/save_concept_names.py --probe_dataset cc12m --probe_split "train"
```
### Download Pre-trained Checkpoints and Concept Names

You can download the necessary checkpoints and associated data from our cloud storage:  
[**Checkpoints and Assigned Names**](https://nextcloud.mpi-klsb.mpg.de/index.php/s/GBFQc3WRP45eY7t)

After downloading, please place the files into the following directory structures:

1. **SAE Checkpoint & Concept Names:**  
   Extract and place inside:  
   `{save_dir_root}/cc12m/dinoclip_ViT-B16/matryoshka/sae_checkpoints/k_12_ef_16_lr_0.0001_mf_[0.008,0.03,0.06,0.12,0.24,0.542]/trainer_0`

2. **Conditional Probability Matrix (D):**  
   Place inside:  
   `{probe_dir}/cc12m/dinoclip_ViT-B16/out/k_12_ef_16_lr_0.0001_mf_[0.008,0.03,0.06,0.12,0.24,0.542]/cc12m/train/Athres_0.001`

3. **Places365 Classification Probe Checkpoint:**  
   Place here:  
   `{probe_dir}/cc12m/dinoclip_ViT-B16/out/k_12_ef_16_lr_0.0001_mf_[0.008,0.03,0.06,0.12,0.24,0.542]/places365/example_linear_probe_places365.pt`

4. **ImageNet Classification Probe Checkpoint:**  
   Place here:  
   `{probe_dir}/cc12m/dinoclip_ViT-B16/out/k_12_ef_16_lr_0.0001_mf_[0.008,0.03,0.06,0.12,0.24,0.542]/imagenet/example_linear_probe_imagenet.pt`
### Quickstart Notebook: CFM and Downstream Tasks
Explore the `cfm_introduction.ipynb` notebook for a hands-on guide on running CFM and applying it to various downstream tasks.
### Visualization
#### Concept Visualization
```
python scripts/visualization/save_top_images_per_concept.py --probe_dataset cc12m --probe_split "train" 
```
#### Hierarchy visualization
```
Coming soon!
```
### Evaluation
```
Coming soon!
```
---
## Citation

If you find this work useful, please cite the arXiv preprint:

```tex
@article{wittenmayer2026cfm,
  title  = {{CFM}: Language-aligned Concept Foundation Model for Vision},
  author = {Wittenmayer, Kai and Rao, Sukrut and Parchami-Araghi, Amin and Schiele, Bernt and Fischer, Jonas},
  journal = {arXiv preprint arXiv:2601.13798},
  year   = {2026}
}
```


