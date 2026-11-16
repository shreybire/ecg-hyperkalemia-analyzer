# Complete Technical Reference: ECG Hyperkalemia Model File-by-File & Workflow Guide

## 1. Project Directory Overview

The project repository located at **`/Users/shreybire/Desktop/Betic/ecgbyshrey2`** consists of 9 core Python modules, CLI scripts, and helper shell scripts:

```
ecgbyshrey2/
├── main.py              # Master CLI entry point & workflow orchestrator
├── lead_cropper.py      # Automated 12-lead to Lead II ROI cropper
├── predict.py           # Single-image inference & Grad-CAM visualizer API
├── transforms.py        # ECG domain data augmentations (NO flips!)
├── dataset.py           # PyTorch Dataset loader & Synthetic Lead-II ECG generator
├── models.py            # Pre-trained ResNet-18 & EfficientNet-B0 backbones
├── losses.py            # Focal Loss & Class-Weighted Cross-Entropy
├── train.py             # Training engine with AdamW & Cosine Annealing
├── metrics.py           # Clinical metrics evaluator (Sensitivity, Specificity, ROC-AUC)
├── gradcam.py           # PyTorch Grad-CAM interpretability engine
├── run_training.sh      # 1-click execution script for model training
└── test_image.sh        # 1-click execution script for single image prediction
```

---

## 2. File-by-File Comprehensive Explanation

### 📄 1. `lead_cropper.py` (Automated Lead II ROI Detector)
* **Purpose**: Automatically extracts the continuous Lead II rhythm strip from 12-lead paper ECG scans/photos.
* **How it Works**:
  1. Inspects image aspect ratio ($w / h$).
  2. If aspect ratio is standard paper format ($1.1 - 1.8$), it auto-crops the **bottom 25% of the page** (where Lead II rhythm strips are standardly printed in clinical 12-lead sheets).
  3. If aspect ratio is already a horizontal single strip ($> 2.2$), it uses the full image without double-cropping.
* **Key Function**: `LeadIICropper().auto_crop_lead_ii(image)`

---

### 📄 2. `transforms.py` (ECG Domain Data Augmentations)
* **Purpose**: Preprocesses and augments images while preserving cardiac electrophysiology.
* **How it Works**:
  * **Resizing**: Scales images to $224 \times 224$ pixels.
  * **STRICT NO FLIPS**: Disables `RandomHorizontalFlip` (reverses time $t$: $T \rightarrow QRS \rightarrow P$) and `RandomVerticalFlip` (reverses voltage $mV$: $ST$ elevation $\rightarrow$ depression).
  * **Safe Transforms**: Allows slight rotation ($\pm 5^\circ$), translation ($\pm 4\%$), scaling ($0.95\times - 1.05\times$), color jitter (brightness/contrast $\pm 15\%$), and Gaussian grid noise.
  * **Normalization**: Normalizes RGB channels using ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
* **Key Function**: `get_ecg_transforms(image_size=(224,224), is_train=True)`

---

### 📄 3. `dataset.py` (Dataset Loader & Synthetic ECG Generator)
* **Purpose**: Loads ECG image files into PyTorch tensors and provides an offline Lead II synthetic ECG generator.
* **How it Works**:
  * `ECGImageDataset`: Inherits from `torch.utils.data.Dataset`. Reads images from `Normal/` (Label 0) and `Hyperkalemia/` (Label 1) subdirectories.
  * `synthesize_lead_ii_signal()`: Generates mathematical Lead II voltage waveforms modeling normal rounded T-waves vs. hyperkalemic **tall tented T-waves** and **flattened P-waves**.
  * `render_ecg_to_image()`: Draws the voltage trace on red/pink ECG grid paper ($25\text{ mm/s}, 10\text{ mm/mV}$).
* **Key Function**: `generate_synthetic_ecg_dataset()`

---

### 📄 4. `models.py` (Neural Network Architectures)
* **Purpose**: Fine-tunes pre-trained Convolutional Neural Networks for binary ECG diagnosis.
* **Supported Backbones**: ResNet-18, ResNet-34, ResNet-50, EfficientNet-B0.
* **Classifier Head Modification**:
  Replaces standard 1000-class ImageNet head with a clinical head:
  $$\text{Dropout}(0.3) \rightarrow \text{Linear}(\text{In}, 128) \rightarrow \text{BatchNorm1d} \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.15) \rightarrow \text{Linear}(128, 2)$$
* **Key Class**: `ECGClassifier`

---

### 📄 5. `losses.py` (Class-Imbalance Loss Functions)
* **Purpose**: Prevents the model from biasing toward majority classes and penalizes dangerous False Negatives.
* **Implemented Losses**:
  1. **Focal Loss ($\gamma=2.0$)**:
     $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
     Focuses gradient updates on hard, misclassified hyperkalemia examples.
  2. **Class-Weighted Cross-Entropy**: Weights classes inversely proportional to dataset frequencies ($w_c = \frac{N}{2 N_c}$).
* **Key Function**: `get_loss_function(loss_type="focal", class_counts=...)`

---

### 📄 6. `train.py` (Training & Validation Loop)
* **Purpose**: Manages model training, learning rate scheduling, validation evaluation, and checkpointing.
* **Features**:
  * **AdamW Optimizer** (`lr=1e-4`, `weight_decay=1e-2`).
  * **Cosine Annealing LR Scheduler** (`T_max=epochs`, `eta_min=1e-6`).
  * **Early Stopping**: Monitors validation loss and saves the best weights checkpoint to `best_ecg_model.pth`.
* **Key Class**: `ECGTrainer`

---

### 📄 7. `metrics.py` (Clinical Evaluation Engine)
* **Purpose**: Computes medical diagnostic metrics and renders diagnostic plots.
* **Calculates**:
  * **Sensitivity / Recall**: $\frac{TP}{TP + FN}$ (Critical: Missed Hyperkalemia Rate)
  * **Specificity**: $\frac{TN}{TN + FP}$ (False Alarm Rate)
  * **Precision (PPV)**: $\frac{TP}{TP + FP}$
  * **F1-Score**: $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
  * **ROC-AUC Score**: Area under Receiver Operating Characteristic curve via trapezoidal integration.
* **Plots**: Renders `confusion_matrix.png` and `roc_curve.png`.
* **Key Class**: `ClinicalMetricsEvaluator`

---

### 📄 8. `gradcam.py` (Explainable AI / Visual Heatmaps)
* **Purpose**: Extracts feature activation maps to visually explain which regions of the ECG the model is focusing on.
* **How it Works**:
  1. Registers forward & backward hooks on the last conv layer (`layer4[-1]` in ResNet-18).
  2. Computes gradient channel weights $\alpha_k = \frac{1}{HW} \sum \frac{\partial y^c}{\partial A^k}$.
  3. Computes Grad-CAM map $L_{\text{Grad-CAM}} = \text{ReLU}\left(\sum \alpha_k A^k\right)$.
  4. Renders a 3-panel figure comparing Raw ECG Image, Grad-CAM Heatmap, and Clinical Overlay.
* **Key Class**: `GradCAM`

---

### 📄 9. `predict.py` (Single-Image Prediction API & CLI)
* **Purpose**: Runs instant inference on any single input ECG image file.
* **Workflow**:
  1. Calls `lead_cropper.py` to auto-crop Lead II.
  2. Passes cropped image to `transforms.py`.
  3. Loads `best_ecg_model.pth` and runs forward pass to get probability percentages.
  4. Generates Grad-CAM overlay and saves 3-panel figure to `results/prediction_result.png`.
* **Key Function**: `predict_single_ecg(image_path="...")`

---

### 📄 10. `main.py` (Master Orchestrator CLI)
* **Purpose**: Provides a unified Command Line Interface (CLI) connecting all modules.
* **Supported Commands**:
  * `--data_dir`: Train on custom dataset folder.
  * `--generate_synthetic`: Run synthetic Lead-II demo.
  * `--checkpoint`: Load saved model for evaluation or Grad-CAM only.
  * `--model_name`: Switch backbone (`resnet18`, `resnet50`, `efficientnet_b0`).
  * `--no_pretrained`: Initialize backbone locally without remote download.

---

## 3. How the Files Connect (Data & Control Flow Diagram)

```
                       [ Input Image File ]
                                │
                                ▼
                       `lead_cropper.py`  (Auto-crops Lead II)
                                │
                                ▼
                        `transforms.py`   (Resize 224x224, No Flips, Normalize)
                                │
                                ▼
                         `dataset.py`     (PyTorch DataLoader)
                                │
                                ▼
                         `models.py`      (ResNet-18 / EfficientNet-B0)
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
          [ Training Mode ]             [ Inference Mode ]
                │                               │
          `losses.py` (Focal Loss)       `predict.py`
                │                               │
          `train.py`  (AdamW + Cosine)        ▼
                │                        `gradcam.py` (3-Panel Overlay)
                ▼                               │
          `metrics.py` (Sens/Spec/AUC)          ▼
                │                    [ Output Visualization ]
                ▼
      [ best_ecg_model.pth ]
```
