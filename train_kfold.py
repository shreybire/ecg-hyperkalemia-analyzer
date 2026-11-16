"""
Stratified K-Fold Cross-Validation Training Script for Genuine ECG Hyperkalemia Classification.

Runs Stratified K-Fold on real clinical ECG images (e.g. from MIMIC / hospital scans).
Evaluates real, uninflated metrics:
- Mean Sensitivity / Recall across K folds (+/- std dev)
- Mean Specificity across K folds (+/- std dev)
- Mean F1-Score across K folds (+/- std dev)
- Mean ROC-AUC score across K folds (+/- std dev)
- Mean Accuracy across K folds (+/- std dev)
"""

import os
import json
import argparse
import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from transforms import get_ecg_transforms
from dataset import ECGImageDataset, create_samples_from_dir
from models import build_model
from losses import get_loss_function
from train import ECGTrainer
from metrics import ClinicalMetricsEvaluator


def run_kfold_cross_validation(
    data_dir: str = "/Users/shreybire/Desktop/Betic/ecgbyshrey2/all 3 combne",
    n_splits: int = 5,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-4,
    model_name: str = "resnet18",
    max_samples_per_class: int = 1000,
    seed: int = 42
):
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"  GENUINE {n_splits}-FOLD CROSS-VALIDATION ON REAL ECGs")
    print(f"=======================================================")
    print(f"Data Directory: {data_dir}")
    print(f"Device: {device.upper()} | Model: {model_name} | Epochs per fold: {epochs} | Batch Size: {batch_size}")

    # Gather all samples from the directory
    all_samples = create_samples_from_dir(data_dir)
    if len(all_samples) == 0:
        for split in ["train", "val", "test"]:
            sub_dir = os.path.join(data_dir, split)
            if os.path.exists(sub_dir):
                all_samples.extend(create_samples_from_dir(sub_dir))

    if len(all_samples) == 0:
        raise FileNotFoundError(f"No ECG images found in '{data_dir}'")

    # Separate by class
    normal_samples = [s for s in all_samples if s[1] == 0]
    hyper_samples = [s for s in all_samples if s[1] == 1]
    print(f"Available Real Images -> Normal: {len(normal_samples)}, Hyperkalemia: {len(hyper_samples)}")

    # Subsample if requested for balanced fast k-fold
    if max_samples_per_class and max_samples_per_class > 0:
        random.shuffle(normal_samples)
        random.shuffle(hyper_samples)
        selected_normal = normal_samples[:max_samples_per_class]
        selected_hyper = hyper_samples[:max_samples_per_class]
        selected_samples = selected_normal + selected_hyper
    else:
        selected_samples = normal_samples + hyper_samples

    random.shuffle(selected_samples)
    print(f"Selected Dataset for Cross-Validation: {len(selected_samples)} total images")
    print(f"  - Class 0 (Normal)       : {sum(1 for s in selected_samples if s[1] == 0)}")
    print(f"  - Class 1 (Hyperkalemia) : {sum(1 for s in selected_samples if s[1] == 1)}")

    # Stratified K-Fold indices split
    labels = np.array([sample[1] for sample in selected_samples])
    indices_class0 = np.where(labels == 0)[0]
    indices_class1 = np.where(labels == 1)[0]

    np.random.shuffle(indices_class0)
    np.random.shuffle(indices_class1)

    folds_c0 = np.array_split(indices_class0, n_splits)
    folds_c1 = np.array_split(indices_class1, n_splits)

    fold_indices = []
    for k in range(n_splits):
        val_idx = np.concatenate([folds_c0[k], folds_c1[k]])
        np.random.shuffle(val_idx)
        train_idx = np.setdiff1d(np.arange(len(selected_samples)), val_idx)
        np.random.shuffle(train_idx)
        fold_indices.append((train_idx, val_idx))

    # Metric accumulators
    fold_results = []
    evaluator = ClinicalMetricsEvaluator()
    train_transform = get_ecg_transforms(is_train=True)
    eval_transform = get_ecg_transforms(is_train=False)

    for fold, (train_idx, val_idx) in enumerate(fold_indices, 1):
        print(f"\n-------------------------------------------------------")
        print(f"  TRAINING FOLD {fold} / {n_splits}")
        print(f"  (Train: {len(train_idx)} [Norm: {sum(labels[train_idx]==0)}, HyperK: {sum(labels[train_idx]==1)}], "
              f"Val: {len(val_idx)} [Norm: {sum(labels[val_idx]==0)}, HyperK: {sum(labels[val_idx]==1)}])")
        print(f"-------------------------------------------------------")

        train_samples = [selected_samples[i] for i in train_idx]
        val_samples = [selected_samples[i] for i in val_idx]

        train_loader = DataLoader(
            ECGImageDataset(train_samples, transform=train_transform),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            ECGImageDataset(val_samples, transform=eval_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        model = build_model(model_name=model_name, num_classes=2, pretrained=False, device=device)
        class_counts = {0: int(np.sum(labels[train_idx] == 0)), 1: int(np.sum(labels[train_idx] == 1))}

        ckpt_name = f"best_model_real_fold_{fold}.pth"
        trainer = ECGTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_counts=class_counts,
            loss_type="focal",
            learning_rate=lr,
            epochs=epochs,
            patience=4,
            checkpoint_path=ckpt_name,
            device=device
        )
        trainer.fit()

        # Evaluate on validation fold
        val_loss, metrics = trainer.validate_epoch()
        fold_results.append({
            "fold": fold,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "val_loss": float(val_loss),
            "accuracy": float(metrics["accuracy"]),
            "sensitivity": float(metrics["sensitivity"]),
            "specificity": float(metrics["specificity"]),
            "precision": float(metrics["precision"]),
            "f1_score": float(metrics["f1_score"]),
            "roc_auc": float(metrics["roc_auc"]),
            "confusion_matrix": metrics["confusion_matrix"].tolist()
        })

        print(f">> Fold {fold} Result: Accuracy = {metrics['accuracy']*100:.2f}%, "
              f"Sens (Recall) = {metrics['sensitivity']*100:.2f}%, "
              f"Spec = {metrics['specificity']*100:.2f}%, "
              f"F1 = {metrics['f1_score']:.4f}, "
              f"ROC-AUC = {metrics['roc_auc']:.4f}")

    # Aggregated Summary
    accs = [r["accuracy"] for r in fold_results]
    sens = [r["sensitivity"] for r in fold_results]
    specs = [r["specificity"] for r in fold_results]
    f1s = [r["f1_score"] for r in fold_results]
    aucs = [r["roc_auc"] for r in fold_results]

    summary = {
        "dataset_source": data_dir,
        "n_splits": n_splits,
        "total_selected_images": len(selected_samples),
        "epochs_per_fold": epochs,
        "batch_size": batch_size,
        "folds": fold_results,
        "summary": {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "mean_sensitivity": float(np.mean(sens)),
            "std_sensitivity": float(np.std(sens)),
            "mean_specificity": float(np.mean(specs)),
            "std_specificity": float(np.std(specs)),
            "mean_f1": float(np.mean(f1s)),
            "std_f1": float(np.std(f1s)),
            "mean_roc_auc": float(np.mean(aucs)),
            "std_roc_auc": float(np.std(aucs))
        }
    }

    # Save summary report to JSON
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "kfold_real_metrics_summary.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "="*70)
    print(f"     GENUINE {n_splits}-FOLD CROSS-VALIDATION SUMMARY REPORT (REAL ECGs)     ")
    print("="*70)
    print(f" Mean Accuracy             : {np.mean(accs)*100:.2f}%  ±  {np.std(accs)*100:.2f}%")
    print(f" Mean Sensitivity (Recall) : {np.mean(sens)*100:.2f}%  ±  {np.std(sens)*100:.2f}%")
    print(f" Mean Specificity          : {np.mean(specs)*100:.2f}%  ±  {np.std(specs)*100:.2f}%")
    print(f" Mean F1-Score             : {np.mean(f1s):.4f}  ±  {np.std(f1s):.4f}")
    print(f" Mean ROC-AUC Score        : {np.mean(aucs):.4f}  ±  {np.std(aucs):.4f}")
    print("="*70)
    print(f"Summary JSON saved to: {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genuine Stratified K-Fold Cross-Validation on Real ECGs.")
    parser.add_argument("--data_dir", type=str, default="/Users/shreybire/Desktop/Betic/ecgbyshrey2/all 3 combne")
    parser.add_argument("--k_splits", type=int, default=5, help="Number of folds (default: 5).")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per fold.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples_per_class", type=int, default=1000, help="Samples per class for balanced fold (default: 1000 -> 2000 total).")
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--lr", type=float, default=1e-4)
    
    args = parser.parse_args()
    run_kfold_cross_validation(
        data_dir=args.data_dir,
        n_splits=args.k_splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_name=args.model_name,
        max_samples_per_class=args.max_samples_per_class
    )
