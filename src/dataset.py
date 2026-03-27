"""
Dataset module for loading and augmenting MedMNIST biomedical image data.

This module provides functions for downloading the PathMNIST dataset,
applying data augmentation, and creating PyTorch DataLoaders.
"""

import medmnist
import numpy as np
import torch
from medmnist import PathMNIST
from torch.utils.data import DataLoader
from torchvision import transforms


# ImageNet normalization statistics used by pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# PathMNIST class names (9-class colorectal tissue classification)
CLASS_NAMES = [
    "Adipose",
    "Background",
    "Debris",
    "Lymphocytes",
    "Mucus",
    "Smooth Muscle",
    "Normal Colon Mucosa",
    "Cancer-Associated Stroma",
    "Colorectal Adenocarcinoma Epithelium",
]


def get_transforms(augment=True, target_size=224):
    """Create data transforms for training and evaluation.

    Images are loaded at 28x28 to save memory, then resized to target_size
    (224x224) for model compatibility.

    Args:
        augment: If True, apply data augmentation (for training).
                 If False, only resize and normalize (for val/test).
        target_size: Target image resolution for the model (default 224).

    Returns:
        torchvision.transforms.Compose object.
    """
    if augment:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((target_size, target_size), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((target_size, target_size), antialias=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return transform


def load_pathmnist(data_dir="./data", image_size=224):
    """Download and load PathMNIST dataset with appropriate transforms.

    Args:
        data_dir: Root directory for storing the dataset.
        image_size: Target image resolution (default 224).

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    import os
    os.makedirs(data_dir, exist_ok=True)

    # Load at 28x28 to save memory, resize to image_size via transforms
    train_transform = get_transforms(augment=True, target_size=image_size)
    eval_transform = get_transforms(augment=False, target_size=image_size)

    train_dataset = PathMNIST(
        split="train", transform=train_transform,
        download=True, root=data_dir, size=28,
    )
    val_dataset = PathMNIST(
        split="val", transform=eval_transform,
        download=True, root=data_dir, size=28,
    )
    test_dataset = PathMNIST(
        split="test", transform=eval_transform,
        download=True, root=data_dir, size=28,
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset,
                       batch_size=32, num_workers=2):
    """Create DataLoaders for train, validation, and test sets.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Test dataset.
        batch_size: Batch size for all loaders.
        num_workers: Number of worker processes for data loading.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_raw_dataset(data_dir="./data", image_size=224):
    """Load the dataset without normalization for visualization purposes.

    Args:
        data_dir: Root directory for storing the dataset.
        image_size: Target image resolution.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset) with ToTensor only.
    """
    import os
    os.makedirs(data_dir, exist_ok=True)

    # Load at 28x28 and resize to target for visualization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size), antialias=True),
    ])

    train_dataset = PathMNIST(
        split="train", transform=transform,
        download=True, root=data_dir, size=28,
    )
    val_dataset = PathMNIST(
        split="val", transform=transform,
        download=True, root=data_dir, size=28,
    )
    test_dataset = PathMNIST(
        split="test", transform=transform,
        download=True, root=data_dir, size=28,
    )

    return train_dataset, val_dataset, test_dataset
