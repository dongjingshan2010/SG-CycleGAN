# SG-CycleGAN
This repository implements SG-CycleGAN for MRI-to-ultrasound synthesis to address data scarcity in endometrial cancer screening.
# SG-CycleGAN: Structure-Guided Cycle-Consistent Adversarial Network for MRI-to-Ultrasound Synthesis

This repository contains the official PyTorch implementation of the paper:

**Efficient Endometrial Carcinoma Screening: Trained on Augmented MRI-to-Ultrasound Images Synthesized by Structure-Guided Cycle-Consistent Adversarial Network**  
*Dongjing Shan, Jiqing Xuan, Yamei Luo, Mengchu Yang, Zeyu Chen, Fajin Lv, Yong Tang, Chunxiang Zhang*

## Overview

SG-CycleGAN addresses data scarcity and class imbalance in endometrial carcinoma screening by generating anatomically faithful ultrasound images from unpaired MRI data. The framework introduces:

- **Modality-Agnostic Feature Extractor (MAFE)** with gradient reversal to preserve shared anatomical structures
- **Feature Consistency Loss** to maintain structural integrity during cross-modal translation
- **Two-stage training strategy** combining SG-CycleGAN with Rectified Flow for high-quality synthesis

When combined with lightweight classifiers (MobileNet-V2, MobileViT), the synthetic data enables diagnostic performance exceeding experienced sonographers while maintaining low computational cost suitable for primary care settings.

## Features

- Unpaired MRI-to-ultrasound image translation
- Structure preservation via modality-agnostic feature extraction
- Multi-GPU distributed training support
- Integrated Rectified Flow for enhanced synthesis quality
- Comprehensive inference pipeline with contrast enhancement

## Data Preparation

Organize your dataset as follows:

```
datasets/
└── Med_shallowdeep/
    ├── trainA/      # MRI images (domain A)
    ├── trainB/      # Ultrasound images (domain B)
    ├── testA/       # MRI test images
    └── testB/       # Ultrasound test images
```

## Training

### Stage 1: SG-CycleGAN Training

```bash
python train.py --dataroot ./datasets/Med_shallowdeep/ \
                --batchSize 6 \
                --n_epochs 30 \
                --lr 0.0002 \
                --cuda
```

**Key Parameters:**
- `--batchSize`: Batch size (default: 6)
- `--n_epochs`: Number of training epochs (default: 30)
- `--lr`: Initial learning rate (default: 0.0002)
- `--mtwg_fea`: MAFE feature dimension (default: 32)
- `--size`: Image crop size (default: 256)

The model automatically trains both SG-CycleGAN and the subsequent Rectified Flow stage. Checkpoints are saved to `output/` directory.

### Distributed Training

For multi-GPU training, the script automatically detects available GPUs and uses distributed data parallel:

```bash
python train.py --cuda --batchSize 12   # Auto-distributes across GPUs
```

## Model Architecture

### SG-CycleGAN Components

- **SharedGenerator**: Contains MAFE and modality-specific mapping functions
- **Discriminators**: Distinguish real vs. synthetic images in each domain
- **Domain Classifier**: Enforces modality-invariant features via gradient reversal

### Loss Functions

- Adversarial Loss (GAN)
- Cycle Consistency Loss
- Modality-Agnostic Feature Consistency Loss
- Identity Loss
- Domain Discrimination Loss
