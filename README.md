# Human-Object Interaction Detection in Fisheye-Distorted Images

## Overview
This repository contains the implementation code for adapting the **HoiTransformer** model—originally trained on the **HICO-DET** dataset—to detect **Human-Object Interactions (HOI)** in **fisheye-distorted images**.

<div align="center">
  <img src="data/architecture.png" width="900px" />
</div>

### Features
- **Transfer Learning Implementation**: Adapting the HoiTransformer model for fisheye images.
- **Inference Pipeline**: Efficient HOI detection and analysis.
- **Visualization Tools**: Methods for analyzing and interpreting model predictions.

This model was fine-tuned using two custom fisheye image datasets, developed in collaboration with **ScaDS.AI** as part of a bachelor’s thesis. Due to privacy constraints, these datasets are **not publicly available**. For access inquiries, please contact the repository maintainer.

---

## System Requirements
- **OS**: Ubuntu 24.04 (recommended)
- **Package Manager**: Miniconda
- **Hardware**: NVIDIA GPU with CUDA support

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/lremane/HoiTransformer.git
cd HoiTransformer
```

### 2. Install Miniconda (if not already installed)
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### 3. Create and Activate the Environment
Please use the provided`environment.yml`to reproduce results:
```bash
conda env create --name HoiTransformer --file environment.yml
conda activate HoiTransformer
```

### 4. Organize Datasets
Ensure your training and testing datasets are structured in the `data/` directory. For transfer learning using the **HICO-DET pretrained model**, follow this structure:
```
HoiTransformer/
├── data/
│   ├── detr_coco/
│   ├── hico/
│   │   ├── eval/
│   │   └── images/
│   │       ├── train/
│   │       └── test/
```
An example dataset structure can be found in the official [HoiTransformer Repository](https://github.com/bbepoch/HoiTransformer).

---

## Training

### 1. Download Pretrained Models
#### ResNet Backbone (Weight Initialization)
```bash
pip install gdown
gdown https://drive.google.com/uc?id=1-WQnnTHB7f7X2NpqPVqIO6tvWN6k1Ot8 -O res50_hico_1cf00bb.pth
```

#### DETR-R50 Transformer (Weight Initialization)
```bash
wget https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth -P data/detr_coco/
```

### 2. Configure Training
Determine the maximum image dimension in your dataset (e.g., `--image_size=1600` for **1600×1600** images).

### 3. Start Training
```bash
python finetune.py \
    --resume=res50_hico_1cf00bb.pth \
    --dataset_file=hico \
    --image_size=1600 \
    --epochs=20 \
    --lr_drop=10 \
    --batch_size=2 \
    --backbone=resnet50 \
    --lr=0.00001 \
    --lr_backbone=0.00001 \
    --dont_use_checkpoint_state
```

---

## Model Evaluation

### 1. Performance Metrics (mAP, Inference Speed)
```bash
python test.py \
    --model_path=model_name.pth \
    --backbone=resnet50 \
    --batch_size=1 \
    --dataset_file=hico \
    --log_dir=./
```

### 2. Result Visualization
```bash
python test_on_images.py \
    --image_directory=<input_directory> \
    --backbone=resnet50 \
    --batch_size=1 \
    --log_dir=<output_directory> \
    --model_path=<model_name>.pth \
    --dataset=hico \
    --hoi_th=0.6 \
    --human_th=0.4 \
    --object_th=0.4 \
    --top_k=5
```

#### Threshold Parameters
- `--hoi_th`: HOI confidence threshold (**default: 0.6**)
- `--human_th`: Human detection confidence threshold (**default: 0.4**)
- `--object_th`: Object detection confidence threshold (**default: 0.4**)
- `--top_k`: Maximum interactions to visualize (**default: 5**)

---

## Acknowledgements
This work builds upon research from: [End-to-End Human Object Interaction Detection with HOI Transformer](https://arxiv.org/abs/2103.04503).

The original implementation can be found at: [HoiTransformer Repository](https://github.com/bbepoch/HoiTransformer).

