"""
Lead II Auto-Cropper & Region of Interest (ROI) Detector for 12-Lead ECG Images.

Clinical Background:
--------------------
Standard 12-lead clinical paper ECG sheets are printed in a 3x4 grid layout with a long
continuous Lead II rhythm strip running along the bottom 20-25% of the page.

This module automatically:
1. Detects whether an image is a full 12-lead ECG page vs. an already isolated single lead strip.
2. Automatically crops the continuous Lead II rhythm strip from 12-lead paper scans.
3. Prepares the cropped Lead II image for PyTorch model inference.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


class LeadIICropper:
    """
    Automated Lead II ECG Waveform Extractor.
    """
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size

    def auto_crop_lead_ii(self, image: Image.Image) -> Tuple[Image.Image, bool, Tuple[int, int, int, int]]:
        """
        Detects layout and automatically crops the Lead II rhythm strip.
        
        Args:
            image (PIL.Image.Image): Input ECG image (full page or single lead)
            
        Returns:
            Tuple[PIL.Image.Image, bool, bbox]:
                - cropped_image: PIL Image of isolated Lead II
                - was_cropped: True if full 12-lead page was detected and cropped; False if already single lead
                - bbox: (xmin, ymin, xmax, ymax) crop coordinates used
        """
        np_img = np.array(image.convert("RGB"))
        h, w, c = np_img.shape
        aspect_ratio = w / float(h)

        # Case 1: Already a single lead strip (long & horizontal aspect ratio > 2.2, or small square)
        if aspect_ratio >= 2.2:
            print("Detected single-lead ECG strip format. Using full image.")
            return image, False, (0, 0, w, h)

        # Case 2: Full 12-Lead Paper Page (Standard 4:3 or A4 aspect ratio 1.1 - 1.8)
        # Standard clinical 12-lead layout puts Lead II continuous rhythm strip in bottom 22% of page
        print("Detected full 12-lead paper ECG sheet layout. Auto-cropping Lead II rhythm strip...")

        # Region of Interest (ROI) for Lead II bottom rhythm strip
        ymin = int(0.70 * h)
        ymax = int(0.96 * h)
        xmin = int(0.04 * w)
        xmax = int(0.96 * w)

        cropped_np = np_img[ymin:ymax, xmin:xmax]

        # Convert back to PIL Image
        cropped_pil = Image.fromarray(cropped_np)
        return cropped_pil, True, (xmin, ymin, xmax, ymax)


def crop_and_save_lead_ii(input_image_path: str, output_image_path: str = "lead_ii_cropped.png") -> str:
    """
    Utility function to crop Lead II from file path and save to disk.
    """
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    pil_img = Image.open(input_image_path)
    cropper = LeadIICropper()
    cropped_img, was_cropped, bbox = cropper.auto_crop_lead_ii(pil_img)
    
    cropped_img.save(output_image_path)
    print(f"Cropped Lead II image saved to: '{output_image_path}' (BBox: {bbox})")
    return output_image_path
