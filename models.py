"""
ECG Classification Model Architectures using Pre-trained PyTorch Backbones.

Supports:
- ResNet-18 / ResNet-34 / ResNet-50
- EfficientNet-B0

Modified with custom dropout & linear classification heads for 2-class ECG diagnosis.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional


class ECGClassifier(nn.Module):
    """
    Fine-tuned Pre-trained Convolutional Neural Network for ECG Binary Classification.
    """
    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False
    ):
        super(ECGClassifier, self).__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes
        
        if self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            # Replace FC layer
            self.backbone.fc = nn.Identity()
            self.target_layer = self.backbone.layer4[-1]
            
        elif self.backbone_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.target_layer = self.backbone.layer4[-1]

        elif self.backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.target_layer = self.backbone.layer4[-1]
            
        elif self.backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            self.target_layer = self.backbone.features[-1]
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose from: resnet18, resnet34, resnet50, efficientnet_b0.")
            
        # Production Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2.0),
            nn.Linear(128, num_classes)
        )
        
        if freeze_backbone:
            self.set_backbone_trainable(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw logits (unnormalized log probabilities).
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def get_target_gradcam_layer(self) -> nn.Module:
        """
        Returns the last convolutional layer used for Grad-CAM feature map extraction.
        """
        return self.target_layer

    def set_backbone_trainable(self, trainable: bool = True):
        """
        Freezes or unfreezes backbone layers for transfer learning.
        """
        for param in self.backbone.parameters():
            param.requires_grad = trainable
        print(f"Backbone parameters set to trainable={trainable}")


def build_model(
    model_name: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
    device: str = "cuda"
) -> ECGClassifier:
    """
    Helper function to instantiate and move ECGClassifier to target device.
    """
    model = ECGClassifier(
        backbone_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout
    )
    return model.to(device)
