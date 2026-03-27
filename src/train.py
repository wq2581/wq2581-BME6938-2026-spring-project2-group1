"""
Training module for biomedical image classification models.

Provides a complete training pipeline with learning rate scheduling,
early stopping, and model checkpointing.
"""

import os
import time
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import yaml

from src.dataset import load_pathmnist, create_dataloaders
from src.models import get_model, count_parameters
from src.evaluate import (
    evaluate_model,
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_history,
)


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch.

    Args:
        model: PyTorch model.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: torch device.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model on the validation set.

    Args:
        model: PyTorch model.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: torch device.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(model_name, config):
    """Full training pipeline for a given model.

    Args:
        model_name: 'custom_cnn' or 'resnet18'.
        config: Configuration dictionary loaded from YAML.

    Returns:
        Tuple of (model, history dict).
    """
    seed = config["training"]["seed"]
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load data
    data_dir = config["dataset"]["data_dir"]
    image_size = config["dataset"]["image_size"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    train_ds, val_ds, test_ds = load_pathmnist(data_dir, image_size)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, batch_size, num_workers
    )

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Create model
    num_classes = config["dataset"]["num_classes"]
    model = get_model(model_name, num_classes=num_classes).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Model: {model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    lr = config["training"]["learning_rate"]
    wd = config["training"]["weight_decay"]
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    num_epochs = config["training"]["num_epochs"]
    patience = config["training"]["patience"]
    results_dir = config["output"]["results_dir"]
    model_dir = config["output"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)

    # Training loop
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(model_dir, f"best_{model_name}.pth")

    print(f"\n{'='*60}")
    print(f"Training {model_name} for up to {num_epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping after {epoch+1} epochs (no improvement for {patience} epochs)")
                break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    print(f"\nLoaded best model from {best_model_path}")

    labels, preds, probs = evaluate_model(model, test_loader, device)
    metrics = compute_metrics(labels, preds, probs, num_classes)

    print(f"\n{'='*60}")
    print(f"Test Results for {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nClassification Report:\n{metrics['classification_report']}")

    # Save plots and metrics
    plot_confusion_matrix(labels, preds, os.path.join(results_dir, f"{model_name}_confusion_matrix.png"))
    plot_roc_curves(labels, probs, num_classes, os.path.join(results_dir, f"{model_name}_roc_curves.png"))
    plot_training_history(history, os.path.join(results_dir, f"{model_name}_training_history.png"))

    # Save metrics to JSON
    metrics_save = {k: v for k, v in metrics.items() if k != "classification_report"}
    metrics_save["classification_report"] = metrics["classification_report"]
    with open(os.path.join(results_dir, f"{model_name}_metrics.json"), "w") as f:
        json.dump(metrics_save, f, indent=2)

    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train biomedical image classification models")
    parser.add_argument("--model", type=str, default="custom_cnn",
                        choices=["custom_cnn", "resnet18", "both"],
                        help="Model to train: custom_cnn, resnet18, or both")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(config["output"]["results_dir"], exist_ok=True)

    if args.model == "both":
        for model_name in ["custom_cnn", "resnet18"]:
            train_model(model_name, config)
    else:
        train_model(args.model, config)


if __name__ == "__main__":
    main()
