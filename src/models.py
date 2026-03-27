"""
Model definitions for biomedical image classification.

Provides two architectures:
1. CustomCNN — a 5-block CNN trained from scratch
2. PretrainedResNet18 — a ResNet-18 fine-tuned with ImageNet weights
"""

import torch
import torch.nn as nn
from torchvision import models


class CustomCNN(nn.Module):
    """A 5-block convolutional neural network trained from scratch.

    Architecture:
        Block 1: Conv(3→32)  → BN → ReLU → MaxPool
        Block 2: Conv(32→64) → BN → ReLU → MaxPool
        Block 3: Conv(64→128) → BN → ReLU → MaxPool
        Block 4: Conv(128→256) → BN → ReLU → MaxPool
        Block 5: Conv(256→512) → BN → ReLU → AdaptiveAvgPool
        Classifier: FC(512→256) → ReLU → Dropout → FC(256→num_classes)
    """

    def __init__(self, num_classes=9):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class PretrainedResNet18(nn.Module):
    """ResNet-18 pretrained on ImageNet, fine-tuned for biomedical classification.

    The final fully connected layer is replaced to match the target number of classes.
    All layers are trainable by default to allow end-to-end fine-tuning.
    """

    def __init__(self, num_classes=9, freeze_backbone=False):
        super().__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.model(x)


def get_model(model_name, num_classes=9):
    """Factory function to create a model by name.

    Args:
        model_name: One of 'custom_cnn' or 'resnet18'.
        num_classes: Number of output classes.

    Returns:
        A PyTorch nn.Module.
    """
    if model_name == "custom_cnn":
        return CustomCNN(num_classes=num_classes)
    elif model_name == "resnet18":
        return PretrainedResNet18(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'custom_cnn' or 'resnet18'.")


def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
