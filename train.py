"""
Training Engine & Validation Loop for ECG Hyperkalemia Classification.

Features:
- AdamW Optimizer
- Cosine Annealing Learning Rate Scheduling
- Class-Weighted Cross-Entropy / Focal Loss for handling class imbalance
- Early Stopping based on Validation Loss / ROC-AUC / Recall
- Automated Best Model Checkpoint Manager
"""

import os
import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Tuple, Optional

from losses import get_loss_function
from metrics import ClinicalMetricsEvaluator


class EarlyStopping:
    """
    Early stopping monitor to prevent model overfitting.
    Tracks validation metric (e.g. val_loss or val_roc_auc).
    """
    def __init__(self, patience: int = 7, delta: float = 1e-4, mode: str = "min"):
        """
        Args:
            patience (int): Number of epochs to wait without improvement before stopping
            delta (float): Minimum change to qualify as an improvement
            mode (str): 'min' for loss, 'max' for metrics like ROC-AUC or Recall
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode.lower()
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return True

        if self.mode == "min":
            improved = current_score < (self.best_score - self.delta)
        else:
            improved = current_score > (self.best_score + self.delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            print(f"  EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class ECGTrainer:
    """
    Production Training and Validation Orchestration.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_counts: Optional[Dict[int, int]] = None,
        loss_type: str = "focal",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        epochs: int = 30,
        patience: int = 7,
        checkpoint_path: str = "best_ecg_model.pth",
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path
        self.device = device
        
        # Loss Function
        self.criterion = get_loss_function(
            loss_type=loss_type,
            class_counts=class_counts,
            gamma=2.0,
            device=device
        )
        
        # Optimizer: AdamW
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # LR Scheduler: Cosine Annealing
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=1e-6
        )
        
        # Early Stopping Monitor
        self.early_stopping = EarlyStopping(patience=patience, mode="min")
        self.evaluator = ClinicalMetricsEvaluator()
        
        self.history = {
            "train_loss": [], "val_loss": [],
            "val_sensitivity": [], "val_specificity": [],
            "val_f1": [], "val_roc_auc": []
        }

    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        
        for images, labels, _ in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    @torch.no_grad()
    def validate_epoch(self) -> Tuple[float, Dict[str, Any]]:
        self.model.eval()
        running_loss = 0.0
        
        all_labels = []
        all_probs = []
        
        for images, labels, _ in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            running_loss += loss.item() * images.size(0)
            
            probs = torch.softmax(logits, dim=1)[:, 1] # Hyperkalemia positive class prob
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
        val_loss = running_loss / len(self.val_loader.dataset)
        metrics = self.evaluator.compute_all_metrics(
            np.array(all_labels),
            np.array(all_probs)
        )
        return val_loss, metrics

    def fit(self) -> Dict[str, Any]:
        """
        Executes full training loop over specified epochs.
        """
        print(f"\nStarting ECG Model Training on {self.device.upper()}...")
        print(f"Total Training Samples  : {len(self.train_loader.dataset)}")
        print(f"Total Validation Samples: {len(self.val_loader.dataset)}")
        print(f"Epochs: {self.epochs} | Initial LR: {self.optimizer.param_groups[0]['lr']}")
        print("-" * 75)
        
        start_time = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.validate_epoch()
            
            # Step learning rate scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_sensitivity"].append(val_metrics["sensitivity"])
            self.history["val_specificity"].append(val_metrics["specificity"])
            self.history["val_f1"].append(val_metrics["f1_score"])
            self.history["val_roc_auc"].append(val_metrics["roc_auc"])
            
            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Sens (Recall): {val_metrics['sensitivity']*100:.1f}% | "
                f"Val Spec: {val_metrics['specificity']*100:.1f}% | "
                f"Val AUC: {val_metrics['roc_auc']:.3f} | LR: {current_lr:.2e}"
            )
            
            # Check for improvement and early stopping
            is_best = self.early_stopping(val_loss)
            if is_best:
                print(f"  >>> Best validation loss achieved ({val_loss:.4f}). Saving checkpoint to '{self.checkpoint_path}'.")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics
                }, self.checkpoint_path)
                best_model_wts = copy.deepcopy(self.model.state_dict())
                
            if self.early_stopping.early_stop:
                print(f"\n[Early Stopping Triggered] Stopping training at epoch {epoch}.")
                break
                
        time_elapsed = time.time() - start_time
        print(f"\nTraining completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        
        # Load best weights
        self.model.load_state_dict(best_model_wts)
        return self.history
