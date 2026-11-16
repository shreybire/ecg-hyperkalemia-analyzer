"""
Grad-CAM (Gradient-Weighted Class Activation Mapping) for ECG Model Interpretability.
Pure OpenCV & PIL implementation (Zero matplotlib dependency).

Provides visual verification allowing clinicians to verify whether the model is focusing on
physiologically relevant features (e.g. T-wave tenting, P-wave flattening, QRS duration)
rather than background grid artifacts or paper borders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageDraw
from typing import Tuple, Optional, List
from transforms import denormalize_ecg_tensor


class GradCAM:
    """
    Self-contained PyTorch Grad-CAM visualizer.
    Registers forward and backward hooks on the last convolutional layer.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        
        if target_layer is None:
            if hasattr(model, "get_target_gradcam_layer"):
                self.target_layer = model.get_target_gradcam_layer()
            else:
                raise ValueError("Target layer for Grad-CAM must be specified.")
        else:
            self.target_layer = target_layer

        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        self.model.zero_grad()
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)
        
        pred_class = logits.argmax(dim=1).item()
        if target_class is None:
            target_class = pred_class
            
        score = probs[0, target_class]
        
        score_target = logits[0, target_class]
        score_target.backward(retain_graph=True)
        
        # Channel weights via Global Average Pooling
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        h_in, w_in = input_tensor.shape[2:]
        cam = F.interpolate(cam, size=(h_in, w_in), mode="bilinear", align_corners=False)
        
        cam_np = cam.squeeze().cpu().numpy()
        cam_max = np.max(cam_np)
        cam_min = np.min(cam_np)
        
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)
            
        return cam_np, pred_class, score.item()


def overlay_gradcam_on_image(
    raw_img_np: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    
    blended = cv2.addWeighted(raw_img_np, 1.0 - alpha, colored_heatmap, alpha, 0)
    return blended


def save_gradcam_visualization(
    input_tensor: torch.Tensor,
    model: nn.Module,
    save_path: str = "gradcam_ecg_sample.png",
    true_label_name: str = "Unknown",
    class_names: List[str] = ["Normal", "Hyperkalemia"]
):
    gradcam = GradCAM(model)
    heatmap, pred_class, score = gradcam.generate_heatmap(input_tensor, target_class=1)
    
    raw_img = denormalize_ecg_tensor(input_tensor)
    overlay = overlay_gradcam_on_image(raw_img, heatmap, alpha=0.45)
    
    # Render heatmap color visual
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Stack 3 panels horizontally (Raw, Heatmap, Overlay)
    h, w, c = raw_img.shape
    combined_img = np.zeros((h, w * 3, c), dtype=np.uint8)
    combined_img[:, :w, :] = raw_img
    combined_img[:, w:w*2, :] = heatmap_color
    combined_img[:, w*2:, :] = overlay
    
    # Save combined interpretability image using PIL/OpenCV
    pil_img = Image.fromarray(combined_img)
    draw = ImageDraw.Draw(pil_img)
    
    pred_label_name = class_names[pred_class]
    draw.text((10, 10), f"Input ECG | True: {true_label_name}", fill=(255, 255, 255))
    draw.text((w + 10, 10), "Grad-CAM Heatmap", fill=(255, 255, 255))
    draw.text((w*2 + 10, 10), f"Clinical Overlay | Pred: {pred_label_name} ({score*100:.1f}%)", fill=(255, 255, 255))
    
    pil_img.save(save_path)
    print(f"Grad-CAM interpretability figure saved to: {save_path}")
