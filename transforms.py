"""
ECG Data Preprocessing and Augmentation Pipeline tailored for ECG Image Classification.

Clinical & Electrophysiological Augmentation Rules:
--------------------------------------------------
1. NO HORIZONTAL FLIP (RandomHorizontalFlip):
   The horizontal axis represents TIME (t). The cardiac cycle strictly follows:
   Atrial Depolarization (P wave) -> Ventricular Depolarization (QRS) -> Ventricular Repolarization (T wave).
   Flipping horizontally reverses temporal order (T wave before P wave), destroying physiological validity.

2. NO VERTICAL FLIP (RandomVerticalFlip):
   The vertical axis represents VOLTAGE / AMPLITUDE (mV).
   Flipping vertically turns ST-elevation into ST-depression, and tall tented T-waves into T-wave inversions.
   This fundamentally alters the diagnostic criteria for hyperkalemia.

3. SAFE AUGMENTATIONS ALLOWED:
   - Slight Rotation (+/- 5 to 7 degrees): Simulates minor document tilt during scanning or photo capture.
   - Translation (+/- 4%): Simulates paper offset within scanner bed or camera frame.
   - Scaling (0.95x - 1.05x): Accounts for distance variations in photo capture without severe distortion.
   - Color / Photometric Jitter: Simulates lighting, contrast, ink darkness, and paper background yellowing.
   - Mild Blur / Noise: Simulates scanner resolution differences and compression artifacts.
"""

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from PIL import Image, ImageEnhance
import random
import numpy as np


class RandomECGGridNoise(object):
    """
    Simulates scanning/paper artifacts such as mild line noise or background grid contrast shifts.
    """
    def __init__(self, p=0.3, noise_std=0.02):
        self.p = p
        self.noise_std = noise_std

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            noise = torch.randn_like(img_tensor) * self.noise_std
            img_tensor = torch.clamp(img_tensor + noise, 0.0, 1.0)
        return img_tensor


def get_ecg_transforms(image_size=(224, 224), is_train=True):
    """
    Returns torchvision transform composition tailored for ECG image classification.
    
    Args:
        image_size (tuple): Target image dimensions (height, width). Default (224, 224).
        is_train (bool): If True, returns training augmentations; else evaluation transforms.
        
    Returns:
        torchvision.transforms.Compose
    """
    # ImageNet normalization standard for pre-trained backbones
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if is_train:
        return T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            # Safe spatial adjustments (strictly preserving cardiac morphology)
            T.RandomRotation(degrees=(-5, 5), fill=255),
            T.RandomAffine(
                degrees=0,
                translate=(0.04, 0.04),
                scale=(0.95, 1.05),
                fill=255
            ),
            # Photometric adjustments for scanner/camera variability
            T.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.1
            ),
            T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
            T.ToTensor(),
            RandomECGGridNoise(p=0.2, noise_std=0.015),
            normalize
        ])
    else:
        return T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            normalize
        ])


def denormalize_ecg_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Utility function to convert normalized PyTorch tensor back to uint8 RGB image for visualization/Grad-CAM.
    
    Args:
        tensor (torch.Tensor): Tensor of shape (C, H, W) or (1, C, H, W) normalized with ImageNet stats.
        
    Returns:
        np.ndarray: Image array of shape (H, W, C) with uint8 values in [0, 255].
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
        
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    
    unnorm = tensor * std + mean
    unnorm = torch.clamp(unnorm, 0.0, 1.0)
    
    np_img = unnorm.cpu().numpy().transpose(1, 2, 0)
    return (np_img * 255).astype(np.uint8)
