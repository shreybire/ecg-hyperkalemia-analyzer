# ECG Hyperkalemia Deep Learning Pipeline: End-to-End Workflow, Clinical Evaluation & Research Paper Blueprint

> **Clinical Summary**  
> Hyperkalemia ($K^+ > 5.5 \text{ mEq/L}$) is a life-threatening electrolyte disorder causing **tall, tented T-waves**, **QRS widening**, and **P-wave flattening** on Lead II ECG traces. This document details the complete system workflow, usage guidelines, clinical strengths, technical limitations, and an academic paper blueprint for research publication.

---

## 1. End-to-End System Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           1. INPUT ECG IMAGE                            │
│  (12-Lead Paper Scan, Camera Photo, 3-Lead Sheet, or Single Strip)     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              2. AUTOMATED LEAD II ROI EXTRACTION ENGINE                 │
│                      (lead_cropper.py)                                  │
│  • Aspect Ratio > 2.2  ──> Single Lead Strip (Use as-is)              │
│  • Aspect Ratio 1.1-1.8 ──> 12-Lead Paper (Auto-crop bottom 25% strip) │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│            3. ELECTROPHYSIOLOGICAL DATA AUGMENTATION                    │
│                      (transforms.py)                                    │
│  • Resizing: (224 x 224) Bilinear Interpolation                         │
│  • STRICT NO FLIPS: Preserves temporal (t) and voltage (mV) direction   │
│  • Safe Transforms: Rotation (±5°), Translation (±4%), Color Jitter     │
│  • ImageNet Normalization: Mean=[0.485,0.456,0.406], Std=[0.229,0.224,0.225]│
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│           4. FINE-TUNED BACKBONE ARCHITECTURE & CLASSIFIER              │
│                      (models.py)                                        │
│  • Backbone: Pre-trained ResNet-18 / EfficientNet-B0                    │
│  • Head: Dropout(0.3) ──> Linear(In, 128) ──> BatchNorm1d ──>           │
│         ReLU ──> Dropout(0.15) ──> Linear(128, 2)                      │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│             5. CLASS-IMBALANCE LOSS & TRAINING ENGINE                   │
│                    (losses.py & train.py)                               │
│  • Loss: Focal Loss (gamma=2.0) with Inverse Frequency Class Weights    │
│  • Optimizer: AdamW (lr=1e-4, weight_decay=1e-2)                        │
│  • LR Scheduler: CosineAnnealingLR (eta_min=1e-6)                       │
│  • Early Stopping: Monitors validation loss & saves best_ecg_model.pth  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│             6. CLINICAL EVALUATION & EXPLAINABLE AI (XAI)               │
│                   (metrics.py & gradcam.py)                             │
│  • Metrics: Sensitivity (Recall), Specificity, F1, ROC-AUC              │
│  • Plots: Confusion Matrix & ROC Curve                                  │
│  • Grad-CAM: 3-Panel Overlay verifying activation on T-Wave Peak       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. How to Use the Model

### A. Single-Image Prediction & Grad-CAM (Instant Diagnostic Test)
To predict on **ANY single ECG image** (full 12-lead scan or single strip):
```bash
cd /Users/shreybire/Desktop/Betic/ecgbyshrey2
"/Users/shreybire/Desktop/Betic/ECG App/backend/.venv/bin/python" predict.py --image "/path/to/ecg_image.png"
```
Or use the 1-click helper script:
```bash
./test_image.sh image.png
```

### B. Full Model Training on Custom Dataset
To train on your dataset with Apple Silicon GPU acceleration:
```bash
python3 main.py --data_dir "/Users/shreybire/Desktop/Betic/all 3 combne" --model_name resnet18 --epochs 15 --loss_type focal
```
Or use the 1-click helper script:
```bash
./run_training.sh
```

### C. Programmatic Python API
```python
from lead_cropper import LeadIICropper
from predict import predict_single_ecg

# Run automated Lead II crop and prediction
pred_label, confidence, output_plot_path = predict_single_ecg(
    image_path="patient_ecg_scan.png",
    checkpoint_path="best_ecg_model.pth"
)
print(f"Diagnosis: {pred_label} ({confidence:.1f}% confidence)")
```

---

## 3. Key Strengths of the Architecture

1. **Electrophysiological Invariance Preservation**:
   - By explicitly disabling horizontal and vertical flips, the pipeline preserves cardiac cycle direction ($P \rightarrow QRS \rightarrow T$) and voltage polarity ($ST$ elevation vs depression).
2. **Automated Lead II Extraction**:
   - Handles full 12-lead paper sheets, 3-lead sheets, and single-lead strips automatically without manual pre-cropping.
3. **High Sensitivity for Clinical Screening**:
   - Employs **Focal Loss ($\gamma = 2.0$)** to penalize False Negatives (missed hyperkalemia), achieving high Sensitivity/Recall to ensure patient safety.
4. **Explainable AI (Grad-CAM Verification)**:
   - Clinicians can visually verify whether activation maps highlight **T-wave tenting and QRS duration** rather than scanner artifacts or paper background text.
5. **100% Local & Offline Deployment**:
   - HIPAA compliant; runs locally on Mac CPU/GPU without sending medical images to external cloud APIs.

---

## 4. Technical Shortcomings & Limitations

1. **Single-Lead (Lead II) Scope**:
   - While Lead II is the primary rhythm lead for T-wave tenting, severe hyperkalemia can also cause P-wave loss and QRS widening in precordial leads ($V_2 - V_4$). Combining multi-lead inputs could provide additional diagnostic redundancy.
2. **Paper Artifact Sensitivity**:
   - Severe paper folds, heavy coffee stains, or extreme skew angles ($> 20^\circ$) can degrade Lead II ROI detection if the grid lines are completely obscured.
3. **Classification vs. Continuous Serum $K^+$ Regression**:
   - The current model performs binary classification (Normal vs. Hyperkalemia). It does not regress exact numerical serum potassium levels (e.g. $6.2 \text{ mEq/L}$).

---

## 5. Research Paper Blueprint & Structure

If you are writing a research paper, thesis, or technical manuscript on this model, structure it as follows:

### 📄 Paper Outline & Sections

#### **Title**
*Automated Screening of Hyperkalemia from Single-Lead ECG Images Using Domain-Constrained CNNs and Explainable Grad-CAM Visualizations*

#### **Abstract**
* **Background:** Hyperkalemia is a life-threatening disorder requiring rapid electrocardiographic detection.
* **Methods:** Fine-tuned ResNet-18 model trained with domain-specific spatial augmentations (preserving temporal and voltage polarity) and Focal Loss ($\gamma=2.0$).
* **Results:** Achieved high Sensitivity, Specificity, and ROC-AUC score, validated with Grad-CAM activation maps focusing on T-wave peak morphology.
* **Conclusion:** Provides an offline-capable, explainable clinical decision support tool.

#### **1. Introduction**
* Clinical significance of serum potassium ($K^+ > 5.5 \text{ mEq/L}$).
* Electrocardiographic signs: Tented T-waves, PR prolongation, QRS widening.
* Challenges with digitizing paper ECG records.

#### **2. Materials and Methods**
* **2.1 Dataset Preparation:** Image dataset distribution (Normal vs. Hyperkalemia).
* **2.2 Automated Lead II ROI Extraction:** Aspect ratio rule and bottom rhythm strip cropping algorithm.
* **2.3 Data Augmentation & Physiological Constraints:** Mathematical justification for excluding horizontal and vertical flips.
* **2.4 Model Architecture:** Pre-trained CNN backbone + custom dropout & linear classification head.
* **2.5 Loss Function:** Focal Loss formula:
  $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

#### **3. Results & Experimental Evaluation**
* Confusion Matrix analysis (TN, FP, FN, TP).
* Sensitivity / Recall, Specificity, F1-Score, and ROC-AUC curve plots.
* Grad-CAM 3-panel visual interpretability figures.

#### **4. Discussion & Clinical Implications**
* Comparison of feature activations against clinical T-wave tenting criteria.
* Edge-computing and HIPAA-compliant offline deployment.

#### **5. Conclusion**
* Summary of findings and future expansion to continuous serum $K^+$ regression.
