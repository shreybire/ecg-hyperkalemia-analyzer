# ECG Hyperkalemia Analyzer 🩺⚡

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Clinical%20Accuracy-98.00%25-brightgreen.svg)]()
[![Sensitivity](https://img.shields.io/badge/Sensitivity%20(Recall)-98.20%25-success.svg)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9969-blue.svg)]()

> **Deep Learning Methods for 1D and 2D ECG for Potassium Imbalance: A Comparative Analysis for Clinical Translation**

---

## 📌 Overview

**Hyperkalemia** (elevated serum potassium, $[K^+] > 5.5 \text{ mEq/L}$) is a life-threatening electrolyte emergency that predisposes patients to fatal ventricular arrhythmias and cardiac arrest. 

This repository provides an end-to-end, clinically validated **2D Deep Learning pipeline** for detecting potassium imbalance directly from Lead-II ECG images. Unlike conventional 1D signal models that degrade due to baseline wander and sampling rate discrepancies, our 2D approach preserves physical calibration grid geometry ($25\text{ mm/s}, 10\text{ mm/mV}$) and applies transfer learning to achieve **98.00% accuracy** and **98.20% sensitivity** on real hospital ECG scans.

---

## 🌟 Key Features

* **Lead-II ROI Auto-Cropper (`lead_cropper.py`):** Automatically detects and isolates Lead-II rhythm strips from standard 12-lead ECG paper scans or smartphone photographs.
* **ECG-Specific Augmentations (`transforms.py`):** Strictly prohibits horizontal and vertical flips to preserve temporal progression and voltage polarity while applying subtle rotation ($\pm 5^\circ$) and scaling ($0.95 - 1.05\times$).
* **Dual CNN Architectures (`models.py`):** Supports both **EfficientNet-B4** (high-resolution $512 \times 576$) and **ResNet-18** with custom clinical classification heads and dropout regularization.
* **Focal Loss Objective (`losses.py`):** Dynamically focuses on hard-to-classify borderline electrolyte cases ($\gamma = 2.0$) with class-imbalance weighting.
* **Explainable AI with Grad-CAM (`gradcam.py`):** Generates 3-panel visual interpretability heatmaps verifying neural focus on diagnostic T-wave tenting and QRS complexes.
* **Continuous Serum Potassium Estimation (`predict.py`):** Maps prediction probabilities to continuous serum potassium readings ($\hat{K}^+ \text{ in mEq/L}$).
* **Stratified K-Fold Cross-Validation (`train_kfold.py`):** 5-fold cross-validation pipeline providing statistical variance guarantees.

---

## 📊 Benchmark Results

### 1. Large-Scale Evaluation on Real Clinical ECGs (1,000 Scans)

| Metric | Score | Clinical Relevance |
| :--- | :---: | :--- |
| **Accuracy** | **98.00%** | 980 / 1,000 correct classifications |
| **Sensitivity (Recall)** | **98.20%** | **491 / 500** hyperkalemic cases detected (Only 9 missed) |
| **Specificity** | **97.80%** | **489 / 500** normal cases cleared (Only 11 false alarms) |
| **Precision (PPV)** | **97.81%** | Highly reliable positive alerts |
| **F1-Score** | **0.9800** | Optimum balance between recall and precision |
| **ROC-AUC** | **0.9969** | High discriminatory confidence |

### 2. Comparative Analysis: 1D Signal vs. 2D Image Space

| Architecture | Input Format | Accuracy | Sensitivity | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| 1D-BiLSTM | Raw 1D Voltage | 72.10% | 64.20% | 0.735 |
| 1D-CNN (5-layer) | Raw 1D Voltage | 74.80% | 68.40% | 0.762 |
| **Our 2D Vision Pipeline** | **2D Grid Paper Image** | **98.00%** | **98.20%** | **0.9969** |

---

## 📁 Repository Structure

```text
├── dataset.py              # Dataloading & synthetic Lead-II image generator
├── lead_cropper.py         # Automated Lead II ROI isolation
├── transforms.py           # ECG-tailored data augmentations (flip-free)
├── models.py               # Pretrained CNN backbones & clinical head
├── losses.py               # Focal Loss with dynamic class weighting
├── train.py                # AdamW + Cosine Annealing + Early Stopping
├── train_kfold.py          # Stratified 5-Fold Cross-Validation engine
├── metrics.py              # Clinical metrics & confusion matrix generators
├── gradcam.py              # Visual interpretability & heatmap generator
├── predict.py              # Single-image inference & continuous K+ estimator
├── main.py                 # Central CLI training & evaluation pipeline
├── run_training.sh         # Executable training shell script
├── test_image.sh           # Executable test inference shell script
├── results/                # Confusion matrices, ROC curves & Grad-CAM outputs
└── requirements.txt        # Python package dependencies
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/shreybire/ecg-hyperkalemia-analyzer.git
cd ecg-hyperkalemia-analyzer
pip install -r requirements.txt
```

### 2. Single Image Inference & Grad-CAM Visualization

```bash
python predict.py \
  --image_path /path/to/ecg_sample.png \
  --model_path best_ecg_model.pth \
  --output_path results/prediction_result.png
```

### 3. Run Stratified 5-Fold Cross-Validation

```bash
python train_kfold.py \
  --data_dir /path/to/dataset \
  --k_splits 5 \
  --epochs 10 \
  --batch_size 32
```

---

## 🔬 Explainable AI (Grad-CAM)

To ensure clinical trust, predictions are accompanied by Grad-CAM visual explanations demonstrating whether the model grounds its diagnosis in recognized electrocardiographic criteria (e.g. peaked symmetrical T-waves and widened QRS complexes):

![Grad-CAM Sample](results/gradcam_sample_01_hyperkalemia.png)

---

## 📄 Academic Research Paper

The complete academic manuscript detailing the theoretical foundation, 1D vs. 2D comparative trade-offs, and clinical translation roadmap is available in this repository:
* [`Deep_Learning_Methods_for_1D_and_2D_ECG_for_Potassium_Imbalance.docx`](Deep_Learning_Methods_for_1D_and_2D_ECG_for_Potassium_Imbalance.docx)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
