# Colorectal Cancer Histopathology Classification with CNNs

Automated 9-class classification of colorectal cancer tissue types using deep learning on the PathMNIST dataset.

## Clinical Context

Colorectal cancer is the third most commonly diagnosed cancer and the second leading cause of cancer-related mortality worldwide. Accurate histopathological tissue classification is essential for diagnosis, treatment planning, and prognosis. This project develops deep learning models to automatically classify colorectal tissue into 9 categories from H&E-stained histopathology patches, potentially assisting pathologists in faster and more consistent slide analysis.

**Target users:** Pathologists and clinical laboratories seeking computer-aided diagnosis tools for colorectal tissue classification.

**Tissue classes:** Adipose, Background, Debris, Lymphocytes, Mucus, Smooth Muscle, Normal Colon Mucosa, Cancer-Associated Stroma, Colorectal Adenocarcinoma Epithelium.

## Quick Start

## Preprocessing Pipeline

The PathMNIST dataset consists of 28×28 RGB histopathology image patches. To ensure compatibility with convolutional neural networks such as ResNet18, all images are resized to 224×224.

For the training dataset, data augmentation techniques are applied to improve model generalization. These include random horizontal and vertical flips, random rotations (±15 degrees), and color jitter to simulate variability in staining conditions.

For validation and testing, only resizing and normalization are applied to ensure consistent evaluation.

All images are normalized using ImageNet mean and standard deviation values, allowing the use of pretrained architectures.

## Reproducing Results

To reproduce the results:

1. Install dependencies:
   pip install -r requirements.txt

2. Train Custom CNN:
   python src/train.py --model custom_cnn

3. Train ResNet18:
   python src/train.py --model resnet18

4. Train both models:
   python src/train.py --model both

5. View results:
   Outputs including metrics, confusion matrices, and ROC curves are saved in the results/ directory.

### Prerequisites

- Python 3.9+
- pip
- GPU recommended (NVIDIA CUDA-compatible), CPU also supported

### Installation

```bash
git clone https://github.com/wq2581/bme6938-2026-spring-project2.git
cd bme6938-2026-spring-project2
pip install -r requirements.txt
```

### Training

```bash
# Train both models (Custom CNN + ResNet-18)
python -m src.train --model both --config configs/config.yaml

# Train only the custom CNN
python -m src.train --model custom_cnn --config configs/config.yaml

# Train only ResNet-18
python -m src.train --model resnet18 --config configs/config.yaml
```

**Expected runtime:** ~20-30 minutes per model on GPU, ~2-3 hours on CPU.

### Usage Guide

1. **Explore the dataset**: Open `notebooks/EDA.ipynb` to review class distributions, sample images, and pixel statistics
2. **Train models**: Run the training script as shown above; the dataset is automatically downloaded
3. **Evaluate and compare**: Open `notebooks/demo.ipynb` to load trained models, run inference, and compare performance

## Data Description

- **Dataset**: [PathMNIST](https://medmnist.com/) from the MedMNIST v2 collection
- **Source**: Derived from the NCT-CRC-HE-100K dataset of colorectal cancer histopathology (Kather et al., 2019)
- **Original resolution**: 28x28 pixels, resized to 224x224 via bilinear interpolation
- **Format**: RGB (3-channel)
- **Splits**: Train (89,996) / Validation (10,004) / Test (7,180)
- **Classes**: 9 tissue types
- **License**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Citation**: Yang et al., "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification," *Scientific Data*, 2023.

The dataset is automatically downloaded via the `medmnist` Python package when training or notebooks are run.

## Methods

### Model Architectures

| Model | Description | Parameters |
|-------|------------|------------|
| **Custom CNN** | 5-block CNN (Conv→BN→ReLU→Pool) with FC classifier, trained from scratch | 1,704,201 |
| **ResNet-18** | ImageNet-pretrained ResNet-18 with fine-tuned classification head | 11,310,153 |

### Data Augmentation

- Random horizontal and vertical flips (p=0.5)
- Random rotation (±15 degrees)
- Color jitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Training Strategy

- **Optimizer:** Adam (lr=0.001, weight_decay=1e-4)
- **LR scheduler:** ReduceLROnPlateau (factor=0.5, patience=3)
- **Early stopping:** patience=5 epochs, checkpointing best model
- **Loss function:** Cross-entropy
- **Batch size:** 32
- **Reproducibility:** Fixed random seed (42)

## Results Summary

### Overall Test Performance

| Metric | Custom CNN | ResNet-18 |
|--------|-----------|-----------|
| **Accuracy** | 0.9496 | 0.9501 |
| **Precision** (weighted) | 0.9505 | 0.9506 |
| **Recall** (weighted) | 0.9496 | 0.9501 |
| **F1-Score** (weighted) | 0.9492 | 0.9496 |
| **ROC-AUC** (weighted) | 0.9960 | 0.9976 |

Both models achieve ~95% accuracy. ResNet-18 shows a slight advantage in ROC-AUC (0.998 vs. 0.996), indicating better probability calibration. The most challenging class for both models is Cancer-Associated Stroma (F1: 0.75-0.80), which is expected due to morphological overlap with Smooth Muscle.

### Generated Artifacts

Results are saved in the `results/` directory:
- `custom_cnn_metrics.json` / `resnet18_metrics.json` — full metrics including per-class report
- `*_confusion_matrix.png` — confusion matrices
- `*_roc_curves.png` — one-vs-rest ROC curves for all 9 classes
- `*_training_history.png` — loss and accuracy curves over epochs

## Project Structure

```
bme6938-2026-spring-project2/
├── README.md                    # Project overview and instructions
├── requirements.txt             # Python dependencies with versions
├── configs/
│   └── config.yaml              # Training hyperparameters and paths
├── src/
│   ├── __init__.py
│   ├── dataset.py               # Data loading, augmentation, and DataLoader creation
│   ├── models.py                # Custom CNN and ResNet-18 model definitions
│   ├── train.py                 # Full training pipeline with LR scheduling and early stopping
│   └── evaluate.py              # Metrics computation, confusion matrix, and ROC curve plotting
├── notebooks/
│   ├── EDA.ipynb                # Exploratory Data Analysis (class distribution, pixel stats, augmentation)
│   └── demo.ipynb               # Model inference demo with comparison and sample predictions
├── report/
│   └── Project2_TeamCRC_Report.md  # Project report (Markdown source)
├── results/                     # Training outputs (metrics, plots, model checkpoints)
│   ├── models/                  # Saved model weights (.pth files)
│   ├── *_metrics.json           # Per-model evaluation metrics
│   └── *.png                    # Visualization figures
└── data/                        # Dataset directory (auto-downloaded, not committed)
```

## Dependencies

See `requirements.txt` for the complete list. Key libraries:

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | >= 2.0.0 | Deep learning framework |
| torchvision | >= 0.15.0 | Pretrained models and transforms |
| medmnist | >= 3.0.0 | Dataset loading |
| scikit-learn | >= 1.3.0 | Evaluation metrics |
| matplotlib | >= 3.7.0 | Visualization |
| seaborn | >= 0.12.0 | Statistical plots |
| pandas | >= 2.0.0 | Data analysis |
| PyYAML | >= 6.0 | Configuration parsing |

## Authors and Contributions

| Name | Role |
|------|------|
| **James Garner** | Data preprocessing, documentation, exploratory data analysis, and report writing |
| **Pascual Jahuey** | Custom CNN architecture design, documentation, and report writing |
| **Jai Raccioppi** | Transfer learning (ResNet-18) implementation, documentation, and report writing |
| **Qing Wang** | Team Communication, Pipeline integration, documentation, evaluation metrics, figure, demo coding, and report writing |

## References

1. Yang, J., Shi, R., Wei, D., et al. "MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification." *Scientific Data*, 2023.
2. Kather, J. N., Krisam, J., Charoentong, P., et al. "Predicting survival from colorectal cancer histology slides using deep learning." *PLOS Medicine*, 2019.
3. He, K., Zhang, X., Ren, S., & Sun, J. "Deep Residual Learning for Image Recognition." *CVPR*, 2016.
4. Sung, H., Ferlay, J., Siegel, R. L., et al. "Global Cancer Statistics 2020." *CA: A Cancer Journal for Clinicians*, 2021.
5. Litjens, G., et al. "A survey on deep learning in medical image analysis." *Medical Image Analysis*, 2017.
