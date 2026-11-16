"""
Clinical Evaluation Metrics & Visual Diagnostic Tooling for ECG ML.
Pure NumPy, PyTorch & OpenCV/PIL implementation (Zero matplotlib / sklearn dependency).

Calculates:
- Confusion Matrix (TN, FP, FN, TP)
- Sensitivity / Recall (Primary Screening Metric)
- Specificity (False Alarm Minimization)
- Positive Predictive Value (Precision)
- F1-Score
- ROC-AUC Curve & Score
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Tuple, Any, List


class ClinicalMetricsEvaluator:
    """
    Evaluates clinical machine learning metrics for binary ECG diagnosis.
    Pure NumPy implementation.
    """
    def __init__(self, class_names: List[str] = ["Normal", "Hyperkalemia"]):
        self.class_names = class_names

    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Computes full suite of clinical evaluation metrics given ground truth labels and positive class probabilities.
        """
        y_true = np.array(y_true, dtype=int)
        y_pred_probs = np.array(y_pred_probs, dtype=float)
        y_pred_binary = (y_pred_probs >= threshold).astype(int)
        
        # Confusion Matrix elements
        tn = int(np.sum((y_true == 0) & (y_pred_binary == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred_binary == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred_binary == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred_binary == 1)))
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        # Clinical Metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # True Negative Rate
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0   # PPV
        
        if precision + sensitivity > 0:
            f1 = 2.0 * (precision * sensitivity) / (precision + sensitivity)
        else:
            f1 = 0.0
            
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
        
        # ROC Curve & AUC Score via trapezoidal rule
        desc_indices = np.argsort(y_pred_probs)[::-1]
        y_sorted = y_true[desc_indices]
        
        tps = np.cumsum(y_sorted)
        fps = np.cumsum(1 - y_sorted)
        
        num_pos = np.sum(y_true == 1)
        num_neg = np.sum(y_true == 0)
        
        if num_pos > 0 and num_neg > 0:
            tpr = np.concatenate([[0.0], tps / num_pos])
            fpr = np.concatenate([[0.0], fps / num_neg])
            # Trapezoidal integration for ROC-AUC (NumPy 2.0+ compatible)
            if hasattr(np, "trapezoid"):
                roc_auc = float(np.trapezoid(tpr, fpr))
            elif hasattr(np, "trapz"):
                roc_auc = float(np.trapz(tpr, fpr))
            else:
                roc_auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))
        else:
            tpr = np.array([0.0, 1.0])
            fpr = np.array([0.0, 1.0])
            roc_auc = 0.5
            
        results = {
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": precision,
            "confusion_matrix": cm,
            "counts": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
            "curve_data": {
                "fpr": fpr,
                "tpr": tpr
            }
        }
        return results

    def print_clinical_report(self, metrics: Dict[str, Any]):
        """
        Formats and prints a clinical evaluation summary table.
        """
        print("\n" + "="*65)
        print("         CLINICAL EVALUATION REPORT - HYPERKALEMIA ECG         ")
        print("="*65)
        print(f" Total Test Samples      : {sum(metrics['counts'].values())}")
        print(f" Sensitivity / Recall    : {metrics['sensitivity']*100:.2f}%  (CRITICAL: Missed HyperK = {(1-metrics['sensitivity'])*100:.2f}%)")
        print(f" Specificity             : {metrics['specificity']*100:.2f}%  (False Alarm Rate = {(1-metrics['specificity'])*100:.2f}%)")
        print(f" Precision (PPV)         : {metrics['precision']*100:.2f}%")
        print(f" F1-Score                : {metrics['f1_score']:.4f}")
        print(f" ROC-AUC Score           : {metrics['roc_auc']:.4f}")
        print(f" Overall Accuracy        : {metrics['accuracy']*100:.2f}%")
        print("-" * 65)
        print(" Confusion Matrix Counts :")
        print(f"   - True Negatives  (Normal correctly identified)       : {metrics['counts']['TN']}")
        print(f"   - False Positives (Normal misclassified as HyperK)    : {metrics['counts']['FP']}")
        print(f"   - False Negatives (HyperK missed - DANGEROUS!)        : {metrics['counts']['FN']}")
        print(f"   - True Positives  (HyperK correctly identified)     : {metrics['counts']['TP']}")
        print("="*65 + "\n")


def plot_confusion_matrix(cm: np.ndarray, save_path: str = "confusion_matrix.png", class_names: List[str] = ["Normal", "Hyperkalemia"]):
    """
    Renders and saves a clinical confusion matrix heatmap using PIL / OpenCV.
    """
    img = Image.new("RGB", (600, 500), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Title
    draw.text((120, 20), "Confusion Matrix - ECG Hyperkalemia", fill=(0, 0, 0))
    
    # Axis labels
    draw.text((220, 460), "Predicted Diagnosis", fill=(0, 0, 0))
    draw.text((20, 240), "True\nLabel", fill=(0, 0, 0))
    
    # Draw Matrix Grid
    cell_w, cell_h = 180, 150
    start_x, start_y = 150, 100
    
    max_val = max(1, cm.max())
    
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            x0 = start_x + j * cell_w
            y0 = start_y + i * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            
            # Fill color proportional to count
            intensity = int(255 - 180 * (val / max_val))
            fill_color = (intensity, intensity, 255)
            
            draw.rectangle([x0, y0, x1, y1], fill=fill_color, outline=(100, 100, 100), width=2)
            
            # Cell text
            text_color = (255, 255, 255) if val > max_val / 2.0 else (0, 0, 0)
            draw.text((x0 + cell_w // 2 - 15, y0 + cell_h // 2 - 10), str(val), fill=text_color)
            
    # Class headers
    draw.text((start_x + 50, start_y - 30), class_names[0], fill=(0, 0, 0))
    draw.text((start_x + cell_w + 40, start_y - 30), class_names[1], fill=(0, 0, 0))
    
    draw.text((start_x - 90, start_y + 60), class_names[0], fill=(0, 0, 0))
    draw.text((start_x - 110, start_y + cell_h + 60), class_names[1], fill=(0, 0, 0))
    
    img.save(save_path)
    print(f"Confusion matrix plot saved to: {save_path}")


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, save_path: str = "roc_curve.png"):
    """
    Renders and saves Receiver Operating Characteristic (ROC) curve plot using PIL / OpenCV.
    """
    w, h = 600, 500
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Title & Legend
    draw.text((150, 20), f"ROC Curve (AUC = {roc_auc:.3f})", fill=(0, 0, 0))
    draw.text((150, 460), "1 - Specificity (False Positive Rate)", fill=(0, 0, 0))
    draw.text((15, 240), "Sensitivity", fill=(0, 0, 0))
    
    # Plotting box
    margin_x, margin_y = 80, 60
    plot_w, plot_h = 480, 360
    
    draw.rectangle([margin_x, margin_y, margin_x + plot_w, margin_y + plot_h], outline=(150, 150, 150), width=2)
    
    # Diagonal baseline
    draw.line([(margin_x, margin_y + plot_h), (margin_x + plot_w, margin_y)], fill=(180, 180, 180), width=1)
    
    # ROC curve line
    points = []
    for f, t in zip(fpr, tpr):
        px = margin_x + int(f * plot_w)
        py = margin_y + plot_h - int(t * plot_h)
        points.append((px, py))
        
    for i in range(len(points) - 1):
        draw.line([points[i], points[i+1]], fill=(43, 92, 143), width=3)
        
    img.save(save_path)
    print(f"ROC curve plot saved to: {save_path}")
