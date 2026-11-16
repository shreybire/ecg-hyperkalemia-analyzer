"""
ECG Dataset Class and Synthetic Lead-II ECG Image Generator.
Provides robust dataloading for real ECG images as well as a synthetic generator
modeling Lead II normal vs. hyperkalemic T-wave tenting and P-wave flattening on grid paper.
"""

import os
import glob
import math
import random
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional


class ECGImageDataset(Dataset):
    """
    PyTorch Dataset for 2-class ECG Image Classification.
    Class 0: Normal
    Class 1: Hyperkalemia
    """
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        """
        Args:
            samples (list): List of tuples (file_path, class_idx)
            transform (callable, optional): Transform pipeline
        """
        self.samples = samples
        self.transform = transform
        self.class_to_idx = {"Normal": 0, "Hyperkalemia": 1}
        self.idx_to_class = {0: "Normal", 1: "Hyperkalemia"}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_path

    def get_class_counts(self) -> Dict[int, int]:
        counts = {0: 0, 1: 0}
        for _, label in self.samples:
            counts[label] += 1
        return counts


def create_samples_from_dir(data_dir: str) -> List[Tuple[str, int]]:
    """
    Scans data directory structured as:
      data_dir/Normal/*.png (or .jpg, .jpeg)
      data_dir/Hyperkalemia/*.png (or .jpg, .jpeg)
    """
    samples = []
    class_mapping = {"normal": 0, "hyperkalemia": 1}
    
    valid_exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    
    for sub_folder in os.listdir(data_dir):
        sub_path = os.path.join(data_dir, sub_folder)
        if not os.path.isdir(sub_path):
            continue
            
        lower_name = sub_folder.lower()
        if "hyperkalemia" in lower_name or "kalemia" in lower_name:
            label = 1
        elif "normal" in lower_name:
            label = 0
        else:
            continue
            
        for ext in valid_exts:
            for file_path in glob.glob(os.path.join(sub_path, ext)):
                samples.append((file_path, label))
                
    return samples


# =====================================================================
# Synthetic ECG Lead II Image Generator (For Pipeline Verification & Demo)
# =====================================================================

def synthesize_lead_ii_signal(num_samples=1000, is_hyperkalemic=False, sampling_rate=500) -> np.ndarray:
    """
    Generates realistic single-lead (Lead II) ECG voltage trace (mV vs time).
    
    Electrocardiographic Differences in Hyperkalemia:
    - Normal: P-wave (~0.15 mV), QRS complex, normal rounded asymmetric T-wave (~0.3 mV).
    - Hyperkalemia: P-wave flattening/loss (< 0.05 mV), QRS prolongation,
      and T-WAVE TENTING (tall, narrow-based, symmetrical peak > 0.8 mV).
    """
    t = np.linspace(0, num_samples / sampling_rate, num_samples) # time in seconds
    hr = 70.0 # Heart rate ~70 bpm
    period = 60.0 / hr
    
    signal = np.zeros_like(t)
    
    # Generate beat cardiac cycles
    num_beats = int(math.ceil(t[-1] / period)) + 1
    
    for beat in range(num_beats):
        t_center = beat * period + 0.3
        
        # P-Wave
        p_offset = t_center - 0.16
        if is_hyperkalemic:
            # Flattened / Absent P wave
            p_amp = 0.03
            p_width = 0.04
        else:
            p_amp = 0.15
            p_width = 0.035
        signal += p_amp * np.exp(-((t - p_offset) ** 2) / (2 * p_width ** 2))
        
        # QRS Complex
        # Q wave
        q_offset = t_center - 0.04
        q_amp = -0.15
        q_width = 0.008
        signal += q_amp * np.exp(-((t - q_offset) ** 2) / (2 * q_width ** 2))
        
        # R wave
        r_offset = t_center
        r_amp = 1.2
        r_width = 0.015 if not is_hyperkalemic else 0.022 # Slight widening in hyperkalemia
        signal += r_amp * np.exp(-((t - r_offset) ** 2) / (2 * r_width ** 2))
        
        # S wave
        s_offset = t_center + 0.04
        s_amp = -0.25
        s_width = 0.01
        signal += s_amp * np.exp(-((t - s_offset) ** 2) / (2 * s_width ** 2))
        
        # T-Wave (Diagnostic Marker)
        t_offset = t_center + 0.22
        if is_hyperkalemic:
            # TENTED T-wave: Tall, narrow-base, symmetric peak
            t_amp = 0.95 + random.uniform(-0.05, 0.1)
            t_width = 0.022 # Narrow base
        else:
            # Normal T-wave: Asymmetric, rounded peak
            t_amp = 0.32 + random.uniform(-0.03, 0.03)
            t_width = 0.05
        signal += t_amp * np.exp(-((t - t_offset) ** 2) / (2 * t_width ** 2))

    # Add subtle baseline wander and high-frequency EMG noise
    baseline_wander = 0.05 * np.sin(2 * np.pi * 0.2 * t)
    noise = np.random.normal(0, 0.015, num_samples)
    
    return signal + baseline_wander + noise


def render_ecg_to_image(voltage_signal: np.ndarray, img_size=(400, 300), lead_label="Lead II") -> Image.Image:
    """
    Renders 1D ECG voltage array into a 2D image formatted like standard medical ECG grid paper.
    Grid Paper format: Red/Pink grid lines (1mm small boxes, 5mm large boxes).
    """
    width, height = img_size
    img = Image.new("RGB", (width, height), (255, 240, 240)) # Soft pink background
    draw = ImageDraw.Draw(img)
    
    # Draw ECG Grid Lines
    # Small grid lines (every 5 pixels)
    for x in range(0, width, 5):
        draw.line([(x, 0), (x, height)], fill=(255, 210, 210), width=1)
    for y in range(0, height, 5):
        draw.line([(0, y), (width, y)], fill=(255, 210, 210), width=1)
        
    # Large grid lines (every 25 pixels - 5mm boxes)
    for x in range(0, width, 25):
        draw.line([(x, 0), (x, height)], fill=(240, 160, 160), width=1)
    for y in range(0, height, 25):
        draw.line([(0, y), (width, y)], fill=(240, 160, 160), width=1)
        
    # Map voltage signal to pixel coordinates
    # Standard 10mm/mV calibration
    mid_y = height / 2.0
    scale_y = (height / 3.5) # scaling factor for voltage
    
    x_coords = np.linspace(20, width - 20, len(voltage_signal))
    y_coords = mid_y - voltage_signal * scale_y
    
    points = list(zip(x_coords, y_coords))
    
    # Draw ECG Trace in dark black/blue ink
    for i in range(len(points) - 1):
        draw.line([points[i], points[i+1]], fill=(10, 10, 25), width=2)
        
    # Add Lead II Annotation
    draw.text((25, 15), f"{lead_label} (25mm/s, 10mm/mV)", fill=(60, 60, 60))
    
    return img


def generate_synthetic_ecg_dataset(output_dir: str, num_normal=100, num_hyperkalemia=50) -> Dict[str, List[Tuple[str, int]]]:
    """
    Generates a full synthetic ECG image dataset split into train/val/test splits.
    Includes class imbalance (e.g. 2:1 ratio of Normal to Hyperkalemia) to test focal/weighted loss.
    """
    splits = {'train': (0.7), 'val': (0.15), 'test': (0.15)}
    samples_by_split = {'train': [], 'val': [], 'test': []}
    
    for class_name, count, is_hk in [("Normal", num_normal, False), ("Hyperkalemia", num_hyperkalemia, True)]:
        all_samples = []
        for i in range(count):
            sig = synthesize_lead_ii_signal(is_hyperkalemic=is_hk)
            img = render_ecg_to_image(sig)
            
            # Decide split
            r = random.random()
            if r < 0.7:
                split = 'train'
            elif r < 0.85:
                split = 'val'
            else:
                split = 'test'
                
            split_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_dir, exist_ok=True)
            
            file_path = os.path.join(split_dir, f"{class_name.lower()}_{i+1:04d}.png")
            img.save(file_path)
            
            label = 1 if is_hk else 0
            samples_by_split[split].append((file_path, label))
            
    print(f"Synthetic ECG Dataset successfully generated at: {output_dir}")
    for split in samples_by_split:
        print(f"  - {split.upper()} set: {len(samples_by_split[split])} samples")
        
    return samples_by_split
