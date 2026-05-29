# EuroSAT Land Cover Classification

CNN classifier for satellite image land cover classification built from scratch in PyTorch, achieving **90% test accuracy** across 10 classes.

## Overview

A custom 3-block CNN trained on the EuroSAT satellite imagery dataset to classify land cover types. Implements early stopping, dataset-specific normalization, and proper train/val/test evaluation.

## Dataset

[EuroSAT](https://github.com/phelber/EuroSAT) — Sentinel-2 satellite imagery dataset
- 27,000 labeled images across 10 land cover classes
- 64×64 pixel RGB patches

## Classes
AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake

## Architecture
Input (3 × 64 × 64)
↓
Conv2d(3→32) + ReLU + MaxPool → 32×32
↓
Conv2d(32→64) + ReLU + MaxPool → 16×16
↓
Conv2d(64→128) + ReLU + MaxPool → 8×8
↓
Flatten → Linear(8192→256) → ReLU → Dropout(0.5)
↓
Linear(256→10) → class scores

## Results

- **Test Accuracy: 90.07%**
- Early stopping triggered at epoch 8 (best model at epoch 5)
- Dataset-specific normalization calculated from training data

![Training Results](training_results.png)

## Requirements

```bash
pip install torch torchvision matplotlib numpy
```

## Usage

```bash
git clone https://github.com/margaretjohn14-alt/Eurosat-cnn-classifier
cd Eurosat-cnn-classifier
python eurosat_classifier_main.py
```

Set `TRAIN = True` in the script to retrain, `TRAIN = False` to load saved model.

## Dataset Split

| Split | Size |
|---|---|
| Train | 18,900 (70%) |
| Validation | 4,050 (15%) |
| Test | 4,050 (15%) |