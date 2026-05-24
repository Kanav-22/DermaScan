# DermaScan — Trustworthy AI Skin Lesion Analyzer

[![HuggingFace](https://img.shields.io/badge/HuggingFace-DermaScan-blue)](https://huggingface.co/spaces/ka22nav/DermaScan)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange)](https://pytorch.org)

**Live Demo: https://huggingface.co/spaces/ka22nav/DermaScan**

> Warning: Screening tool only. Not a substitute for professional medical diagnosis.

## What it does
Upload a photo of a skin lesion. DermaScan returns:
- Malignant / Benign prediction
- Grad-CAM heatmap showing where the model focused
- ABCD clinical scores (Asymmetry, Border, Color, Diameter)
- Uncertainty estimate via Monte Carlo Dropout

## Why this project
Existing melanoma classifiers are black boxes. This project implements a full XAI pipeline aligned with the ABCD clinical rule used by dermatologists. Inspired by NTU Singapore GCF Research Project 229: Trustworthy Deep Learning for Melanoma Diagnosis Using the ABCD Rule.

## Model Performance
Trained on HAM10000 (10,015 dermoscopy images).

| Metric | Score |
|---|---|
| AUC-ROC | 0.9294 |
| Melanoma Recall | 80.8% |
| Accuracy | 88.4% |
| F1 Score | 0.61 |

Recall is the primary metric. Missing a melanoma is far more dangerous than a false alarm. Clinical dermatologists typically achieve AUC ~0.86.

## XAI Pipeline

### 1. Grad-CAM (Visual Explanation)
Gradient-weighted Class Activation Mapping highlights which pixels drove the model's decision. Red = high attention, blue = low attention.

### 2. ABCD Classical Scoring (Clinical Explanation)
Parallel OpenCV pipeline scores each criterion independently:
- A - Asymmetry: Flip-based symmetry comparison
- B - Border: Circularity deviation from perfect circle
- C - Color: K-means color cluster variance
- D - Diameter: Lesion area relative to image

### 3. Monte Carlo Dropout (Uncertainty Estimation)
Dropout kept active during inference. 30 forward passes produce a distribution of predictions. Standard deviation = uncertainty. High uncertainty means recommend dermatologist review regardless of prediction.

## Tech Stack

| Component | Technology |
|---|---|
| Model | EfficientNet B0 (PyTorch) |
| Explainability | Grad-CAM + Monte Carlo Dropout |
| Classical CV | OpenCV (ABCD scoring) |
| Frontend | Gradio |
| Deployment | HuggingFace Spaces |
| Dataset | HAM10000 |

## Training Details
- Class imbalance: WeightedRandomSampler + BCEWithLogitsLoss pos_weight=8
- Augmentation: RandomFlip, RandomRotation, ColorJitter
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR
- Best epoch: 25 | Val AUC: 0.9301

## Author
Kanav Behl — BE Electronics & Computer Engineering, TIET Patiala
GitHub: https://github.com/Kanav-22
Live App: https://huggingface.co/spaces/ka22nav/DermaScan
