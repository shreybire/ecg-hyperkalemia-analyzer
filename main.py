"""
Production Master Pipeline Script for ECG Hyperkalemia Classification & Grad-CAM Interpretability.

Usage Examples:
---------------
1. Run complete pipeline on Synthetic Lead-II Dataset (Auto-generated demo):
   python main.py --generate_synthetic --epochs 10 --loss_type focal

2. Train on custom ECG dataset directory:
   python main.py --data_dir /path/to/ecg_dataset --model_name resnet18 --epochs 25

3. Generate Grad-CAM visualizations only from a trained checkpoint:
   python main.py --data_dir /path/to/ecg_dataset --checkpoint best_ecg_model.pth --explain_only
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from transforms import get_ecg_transforms
from dataset import ECGImageDataset, create_samples_from_dir, generate_synthetic_ecg_dataset
from models import build_model
from train import ECGTrainer
from metrics import ClinicalMetricsEvaluator, plot_confusion_matrix, plot_roc_curve
from gradcam import save_gradcam_visualization


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Production Pipeline: ECG Hyperkalemia Screening & Interpretability")
    parser.add_argument("--data_dir", type=str, default="/Users/shreybire/Desktop/Betic/all 3 combne", help="Directory containing train/val/test split or class subfolders.")
    parser.add_argument("--generate_synthetic", action="store_true", help="Generate synthetic Lead-II ECG image dataset for testing.")
    parser.add_argument("--model_name", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0"], help="Backbone CNN architecture.")
    parser.add_argument("--no_pretrained", action="store_true", help="Initialize backbone without downloading ImageNet weights.")
    parser.add_argument("--image_size", type=int, default=224, help="Input image spatial size (height/width).")
    parser.add_argument("--batch_size", type=int, default=16, help="DataLoader batch size.")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate for AdamW.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW weight decay penalty.")
    parser.add_argument("--loss_type", type=str, default="focal", choices=["focal", "weighted_ce", "ce"], help="Loss function to handle class imbalance.")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience epochs.")
    parser.add_argument("--checkpoint", type=str, default="best_ecg_model.pth", help="Path to save/load model checkpoint.")
    parser.add_argument("--explain_only", action="store_true", help="Skip training and run Grad-CAM evaluation on checkpoint.")
    parser.add_argument("--num_explain_samples", type=int, default=4, help="Number of test ECG images to generate Grad-CAM overlays for.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save evaluation plots and Grad-CAM visualizations.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Executing ECG Pipeline on Device: {device.upper()}")
    
    # -----------------------------------------------------------------
    # Step 1: Dataset Preparation
    # -----------------------------------------------------------------
    if args.generate_synthetic:
        print("\n--- Generating Synthetic Lead-II ECG Image Dataset ---")
        generate_synthetic_ecg_dataset(args.data_dir, num_normal=140, num_hyperkalemia=70)

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    test_dir = os.path.join(args.data_dir, "test")

    # Check if standard train/val/test subdirectories exist
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        train_samples = create_samples_from_dir(train_dir)
        val_samples = create_samples_from_dir(val_dir)
        test_samples = create_samples_from_dir(test_dir) if os.path.exists(test_dir) else val_samples
    else:
        # Single directory split
        all_samples = create_samples_from_dir(args.data_dir)
        if len(all_samples) == 0:
            raise FileNotFoundError(
                f"No ECG images found in '{args.data_dir}'. Ensure subfolders 'Normal' and 'Hyperkalemia' exist, "
                "or pass '--generate_synthetic' to auto-generate a sample dataset."
            )
        np.random.shuffle(all_samples)
        n_total = len(all_samples)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        train_samples = all_samples[:n_train]
        val_samples = all_samples[n_train:n_train+n_val]
        test_samples = all_samples[n_train+n_val:]

    train_transform = get_ecg_transforms(image_size=(args.image_size, args.image_size), is_train=True)
    eval_transform = get_ecg_transforms(image_size=(args.image_size, args.image_size), is_train=False)

    train_dataset = ECGImageDataset(train_samples, transform=train_transform)
    val_dataset = ECGImageDataset(val_samples, transform=eval_transform)
    test_dataset = ECGImageDataset(test_samples, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    class_counts = train_dataset.get_class_counts()
    print(f"\nTrain set distribution: Normal (0) = {class_counts[0]}, Hyperkalemia (1) = {class_counts[1]}")

    # -----------------------------------------------------------------
    # Step 2: Model Instantiation
    # -----------------------------------------------------------------
    model = build_model(
        model_name=args.model_name,
        num_classes=2,
        pretrained=not args.no_pretrained,
        dropout=0.3,
        device=device
    )

    # -----------------------------------------------------------------
    # Step 3: Training & Validation Loop
    # -----------------------------------------------------------------
    if not args.explain_only:
        trainer = ECGTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_counts=class_counts,
            loss_type=args.loss_type,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            checkpoint_path=args.checkpoint,
            device=device
        )
        trainer.fit()
    else:
        print(f"\nLoading model checkpoint from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # -----------------------------------------------------------------
    # Step 4: Final Clinical Evaluation on Test Set
    # -----------------------------------------------------------------
    print("\n--- Running Final Clinical Evaluation on Test Set ---")
    model.eval()
    y_true = []
    y_probs = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            y_true.append(labels.item())
            y_probs.append(probs.item())

    evaluator = ClinicalMetricsEvaluator()
    metrics = evaluator.compute_all_metrics(np.array(y_true), np.array(y_probs))
    evaluator.print_clinical_report(metrics)

    # Save Diagnostic Plots
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    roc_path = os.path.join(args.output_dir, "roc_curve.png")
    plot_confusion_matrix(metrics["confusion_matrix"], save_path=cm_path)
    plot_roc_curve(metrics["curve_data"]["fpr"], metrics["curve_data"]["tpr"], metrics["roc_auc"], save_path=roc_path)

    # -----------------------------------------------------------------
    # Step 5: Grad-CAM Interpretability Overlay Generation
    # -----------------------------------------------------------------
    print(f"\n--- Generating Grad-CAM Interpretability Visualizations ({args.num_explain_samples} samples) ---")
    explained_count = 0
    
    for i, (images, labels, img_paths) in enumerate(test_loader):
        if explained_count >= args.num_explain_samples:
            break
            
        img_tensor = images.to(device)
        true_label_idx = labels.item()
        true_name = "Hyperkalemia" if true_label_idx == 1 else "Normal"
        
        save_file = os.path.join(args.output_dir, f"gradcam_sample_{i+1:02d}_{true_name.lower()}.png")
        save_gradcam_visualization(
            input_tensor=img_tensor,
            model=model,
            save_path=save_file,
            true_label_name=true_name
        )
        explained_count += 1

    print(f"\nPipeline Execution Complete! Diagnostic plots and Grad-CAM overlays saved to: '{args.output_dir}'")


if __name__ == "__main__":
    main()