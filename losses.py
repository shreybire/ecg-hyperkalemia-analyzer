"""
Loss Functions for Imbalanced Biomedical Classification.

Provides:
1. Focal Loss (Lin et al., 2017): Focuses learning on hard negative/positive examples.
2. Class-Weighted Cross Entropy: Adjusts loss penalty inversely proportional to class prevalence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union


class FocalLoss(nn.Module):
    r"""
    Focal Loss for addressing class imbalance in binary/multiclass classification.
    
    Formula:
        FL(p_t) = - \alpha_t (1 - p_t)^\gamma \log(p_t)
        
    Args:
        alpha (float, list, or Tensor): Class weights. If float, represents weight for class 1.
        gamma (float): Focusing parameter (typically gamma=2.0). Higher values reduce loss for easy samples.
        reduction (str): 'mean', 'sum', or 'none'.
    """
    def __init__(
        self,
        alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([1.0 - alpha, alpha])
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha)
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha
        else:
            raise TypeError("Alpha must be float, list, or torch.Tensor")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (Tensor): Predicted raw logit values of shape (N, C)
            targets (Tensor): Ground truth class indices of shape (N)
        """
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        
        # Gather target log probabilities and probabilities
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Calculate focal weights: (1 - p_t)^gamma
        focal_weight = (1.0 - target_probs) ** self.gamma
        
        loss = -focal_weight * target_log_probs
        
        # Apply class weights alpha_t if specified
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            alpha_weight = self.alpha.gather(0, targets)
            loss = alpha_weight * loss
            
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def get_loss_function(
    loss_type: str = "focal",
    class_counts: Optional[dict] = None,
    gamma: float = 2.0,
    device: str = "cuda"
) -> nn.Module:
    """
    Factory function to build class-balanced loss functions.
    
    Args:
        loss_type (str): 'focal', 'weighted_ce', or 'ce'
        class_counts (dict): Dictionary {0: count_normal, 1: count_hyperkalemia}
        gamma (float): Focusing parameter for Focal Loss
        device (str): Device to place class weight tensors
        
    Returns:
        nn.Module criterion
    """
    weights = None
    if class_counts is not None and sum(class_counts.values()) > 0:
        total = sum(class_counts.values())
        # Inverse class frequency weighting
        w0 = total / (2.0 * max(1, class_counts.get(0, 1)))
        w1 = total / (2.0 * max(1, class_counts.get(1, 1)))
        weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
        # Normalize weights
        weights = weights / weights.sum()
        print(f"Class frequencies: {class_counts} -> Computed Loss Weights: Normal={weights[0]:.3f}, Hyperkalemia={weights[1]:.3f}")

    if loss_type.lower() == "focal":
        return FocalLoss(alpha=weights, gamma=gamma, reduction="mean")
    elif loss_type.lower() == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weights)
    elif loss_type.lower() == "ce":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Choose from 'focal', 'weighted_ce', or 'ce'.")
