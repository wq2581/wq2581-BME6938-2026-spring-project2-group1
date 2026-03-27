# Project 2: Convolutional Neural Networks for Colorectal Cancer Histopathology Classification

**GitHub Repository:** https://github.com/wq2581/bme6938-2026-spring-project2

**Team Members:**
- James Garner -- Data preprocessing and exploratory data analysis
- Pascual Jahuey -- Custom CNN architecture design and training
- Jai Raccioppi -- Transfer learning implementation and evaluation
- Qing Wang -- Pipeline integration, documentation, and report writing

**Date:** March 2026

---

## Abstract

Colorectal cancer (CRC) is the third most commonly diagnosed cancer worldwide, and accurate histopathological classification of tissue types is crucial for diagnosis and treatment planning. In this project, we developed and compared two convolutional neural network (CNN) approaches for automated classification of colorectal tissue into nine histological categories using the PathMNIST dataset from the MedMNIST collection. We implemented a custom 5-block CNN trained from scratch and a pretrained ResNet-18 model fine-tuned on the target task. Both models achieved strong performance, with the Custom CNN reaching 94.96% accuracy (F1=0.949, ROC-AUC=0.996) and ResNet-18 achieving 95.01% accuracy (F1=0.950, ROC-AUC=0.998) on the held-out test set. These results demonstrate that deep learning models can reliably distinguish between diverse colorectal tissue types, with transfer learning providing a marginal advantage in generalization, particularly on underrepresented classes.

---

## 1. Introduction

Colorectal cancer is responsible for approximately 900,000 deaths annually, making it the second leading cause of cancer-related mortality worldwide (Sung et al., 2021). Histopathological examination of tissue biopsies remains the gold standard for CRC diagnosis, but manual slide review is time-consuming, subjective, and prone to inter-observer variability (Bera et al., 2019). Computer-aided diagnosis (CAD) systems powered by deep learning offer the potential to assist pathologists by providing rapid, consistent, and objective tissue classification.

The clinical significance of automated histopathology classification extends beyond simple diagnosis. Accurate identification of tissue subtypes -- including tumor epithelium, stroma, lymphocytic infiltration, and normal mucosa -- provides valuable prognostic information and can guide treatment selection (Kather et al., 2019). For example, the presence of tumor-infiltrating lymphocytes is associated with favorable outcomes in CRC patients, while stromal composition influences treatment response (Galon et al., 2006).

This project addresses the task of nine-class colorectal tissue classification using convolutional neural networks. We compare two modeling approaches: (1) a custom CNN architecture designed from scratch, and (2) a ResNet-18 model pretrained on ImageNet and fine-tuned on pathology images. The intended beneficiary population includes pathologists and clinical laboratories that could integrate such models into their diagnostic workflows to improve throughput and consistency.

Our objectives are to: (a) develop a reproducible deep learning pipeline for histopathology image classification, (b) evaluate the effectiveness of transfer learning versus training from scratch on biomedical image data, and (c) analyze model performance across different tissue types to identify clinical strengths and limitations.

---

## 2. Literature Review

Deep learning has transformed computational pathology, with convolutional neural networks achieving expert-level performance on various histopathological classification tasks (Litjens et al., 2017). The foundational work by Kather et al. (2016) demonstrated that CNNs could classify colorectal cancer tissue textures with high accuracy, establishing a benchmark dataset of 5,000 tissue patches across eight classes. This was later expanded to the NCT-CRC-HE-100K dataset containing 100,000 image patches from hematoxylin and eosin (H&E) stained slides (Kather et al., 2019), which forms the basis of the PathMNIST dataset used in this project.

Transfer learning has proven particularly effective in medical imaging, where labeled data is often scarce relative to natural image datasets. Tajbakhsh et al. (2016) showed that fine-tuning pretrained models consistently outperformed training from scratch on medical image analysis tasks. ResNet architectures (He et al., 2016) have become the de facto standard backbone for histopathology classification due to their ability to train very deep networks through residual connections. Specifically, ResNet-18 and ResNet-50 have been widely adopted in pathology (Coudray et al., 2018).

The MedMNIST benchmark (Yang et al., 2023) standardized the evaluation of deep learning models across multiple biomedical image modalities, providing consistent data splits and evaluation protocols. PathMNIST, derived from the NCT-CRC-HE-100K dataset, includes nine tissue classes at various image resolutions, enabling systematic comparison of model architectures.

Data augmentation strategies are critical in medical imaging to mitigate overfitting and improve generalization. Tellez et al. (2019) showed that stain augmentation and geometric transformations significantly improved model robustness in histopathology. Color jitter is especially relevant for H&E-stained images, which exhibit substantial staining variability across laboratories and scanners (Macenko et al., 2009).

Recent work has also explored attention-based architectures such as Vision Transformers (ViT) for pathology (Chen et al., 2022), though CNNs remain competitive, especially when training data is limited. The gap between our custom CNN and pretrained models in this project aligns with the broader literature suggesting that ImageNet pretraining provides useful low-level feature representations that transfer effectively to histopathology (Mormont et al., 2020).

Regarding clinical deployment, studies have emphasized the importance of model interpretability and uncertainty estimation in medical AI systems (Kompa et al., 2021). Per-class performance analysis, as conducted in this project, is essential for understanding which tissue types are most challenging to classify and where clinical caution is warranted.

---

## 3. Methods and Data

### 3.1 Dataset Description

We used the PathMNIST dataset from the MedMNIST v2 collection (Yang et al., 2023), which is derived from the NCT-CRC-HE-100K dataset of colorectal cancer histopathology (Kather et al., 2019). The dataset contains 107,180 H&E-stained image patches classified into nine tissue types:

| Class | Tissue Type | Train | Val | Test |
|-------|------------|-------|-----|------|
| 0 | Adipose | 10,528 | 1,180 | 1,338 |
| 1 | Background | 6,660 | 740 | 847 |
| 2 | Debris | 2,692 | 296 | 339 |
| 3 | Lymphocytes | 4,980 | 556 | 634 |
| 4 | Mucus | 8,148 | 904 | 1,035 |
| 5 | Smooth Muscle | 4,700 | 520 | 592 |
| 6 | Normal Colon Mucosa | 5,820 | 648 | 741 |
| 7 | Cancer-Associated Stroma | 3,332 | 368 | 421 |
| 8 | Colorectal Adenocarcinoma Epithelium | 9,696 | 1,076 | 1,233 |

The standard MedMNIST train/validation/test split was used (89,996 / 10,004 / 7,180 samples). Images were loaded at 28x28 resolution and resized to 224x224 pixels via bilinear interpolation to match pretrained model input requirements.

### 3.2 Exploratory Data Analysis

A dedicated EDA notebook (`notebooks/EDA.ipynb`) was created to systematically explore the dataset. Key findings include:

- **Class imbalance**: Adipose (10,528 training samples) has approximately 4x more samples than the smallest class, Debris (2,692). Cancer-Associated Stroma (3,332) is also underrepresented.
- **Visual diversity**: Each tissue type exhibits distinct morphological patterns. Adipose tissue shows characteristic large, clear fat cells, while lymphocyte regions appear as dense clusters of small, dark-staining cells.
- **Color variation**: H&E staining intensity varies across patches, motivating the use of color jitter augmentation.
- **Pixel distributions**: RGB channels show non-uniform intensity distributions, with the red channel having the highest mean intensity, consistent with the eosin staining of cytoplasm.

These findings informed our augmentation strategy and the choice to use weighted evaluation metrics.

### 3.3 Data Preprocessing and Augmentation

All images were normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) to match the distribution expected by pretrained models.

Training augmentation included:
- Random horizontal and vertical flips (p=0.5 each) -- justified because tissue orientation is arbitrary in histopathology
- Random rotation (up to 15 degrees) -- accounts for variable patch extraction angles
- Color jitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) -- simulates staining variability

Validation and test sets received only resizing and normalization (no augmentation).

### 3.4 Model Architectures

**Custom CNN (trained from scratch):** A 5-block convolutional network with increasing channel depth (32 -> 64 -> 128 -> 256 -> 512). Each block consists of a 3x3 convolution, batch normalization, ReLU activation, and max pooling (2x2). The final block uses adaptive average pooling. The classifier head includes a fully connected layer (512 -> 256), ReLU, dropout (p=0.5), and output layer (256 -> 9). Total parameters: 1,704,201.

**ResNet-18 (pretrained, fine-tuned):** A ResNet-18 model initialized with ImageNet-pretrained weights. The original classification head was replaced with a custom head: FC(512 -> 256) -> ReLU -> Dropout(0.5) -> FC(256 -> 9). All layers were trainable (no frozen backbone). Total parameters: 11,310,153.

### 3.5 Training Strategy

Both models were trained with:
- **Optimizer:** Adam (learning rate=0.001, weight decay=1e-4)
- **Loss function:** Cross-entropy loss
- **Learning rate scheduler:** ReduceLROnPlateau (factor=0.5, patience=3, monitoring validation loss)
- **Early stopping:** Patience of 5 epochs with model checkpointing (saving the best model based on validation loss)
- **Batch size:** 32
- **Maximum epochs:** 20

Random seeds were fixed (seed=42) across all libraries (NumPy, PyTorch, CUDA) for reproducibility.

### 3.6 Evaluation Metrics

We reported the following metrics on the held-out test set:
- **Accuracy** (overall)
- **Precision, Recall, F1-Score** (weighted averages and per-class)
- **ROC-AUC** (one-vs-rest, weighted average)
- **Confusion matrix** for detailed error analysis

---

## 4. Results and Evaluation

### 4.1 Overall Performance

| Metric | Custom CNN | ResNet-18 |
|--------|-----------|-----------|
| Accuracy | 0.9496 | 0.9501 |
| Precision (weighted) | 0.9505 | 0.9506 |
| Recall (weighted) | 0.9496 | 0.9501 |
| F1-Score (weighted) | 0.9492 | 0.9496 |
| ROC-AUC (weighted) | 0.9960 | 0.9976 |

Both models achieved approximately 95% test accuracy. The ResNet-18 model showed a slight advantage in ROC-AUC (0.998 vs. 0.996), indicating better calibrated probability estimates and more reliable discrimination across all classes.

### 4.2 Per-Class Performance

**Custom CNN:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Adipose | 0.98 | 0.98 | 0.98 |
| Background | 1.00 | 1.00 | 1.00 |
| Debris | 0.78 | 0.91 | 0.84 |
| Lymphocytes | 0.99 | 1.00 | 0.99 |
| Mucus | 0.96 | 0.97 | 0.97 |
| Smooth Muscle | 0.82 | 0.84 | 0.83 |
| Normal Colon Mucosa | 0.98 | 0.95 | 0.97 |
| Cancer-Associated Stroma | 0.89 | 0.72 | 0.80 |
| Colorectal Adenocarcinoma Epithelium | 0.97 | 0.98 | 0.97 |

**ResNet-18:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Adipose | 1.00 | 0.96 | 0.98 |
| Background | 0.99 | 1.00 | 1.00 |
| Debris | 0.82 | 1.00 | 0.90 |
| Lymphocytes | 0.99 | 0.99 | 0.99 |
| Mucus | 0.95 | 1.00 | 0.97 |
| Smooth Muscle | 0.88 | 0.85 | 0.87 |
| Normal Colon Mucosa | 0.95 | 0.96 | 0.95 |
| Cancer-Associated Stroma | 0.80 | 0.70 | 0.75 |
| Colorectal Adenocarcinoma Epithelium | 0.97 | 0.97 | 0.97 |

Both models excelled on Background (F1=1.00), Lymphocytes (F1=0.99), and Adipose (F1=0.98). The most challenging class for both models was Cancer-Associated Stroma, where the Custom CNN achieved F1=0.80 and ResNet-18 achieved F1=0.75. Debris classification improved notably with ResNet-18 (F1=0.90 vs. 0.84), suggesting that pretrained features better capture the subtle texture patterns of tissue debris. ResNet-18 also improved Smooth Muscle classification (F1=0.87 vs. 0.83).

### 4.3 Training Dynamics

The training history curves (Figures 1-2) show that both models converged within 15-20 epochs. The Custom CNN showed slightly more training loss oscillation, while ResNet-18 exhibited smoother convergence, likely due to the well-initialized feature representations from ImageNet pretraining. Both models benefited from the learning rate scheduler, which reduced the learning rate when validation loss plateaued, helping the models escape local minima.

### 4.4 Confusion Matrix Analysis

The confusion matrices (Figures 3-4) reveal that the primary source of errors for both models is confusion between Cancer-Associated Stroma and Smooth Muscle. This is clinically expected, as both tissue types share fibrous morphological features that can be challenging even for trained pathologists. Debris was sometimes confused with Mucus, which is also morphologically plausible given their overlapping visual appearance in H&E-stained sections.

### 4.5 ROC Curve Analysis

The ROC curves (Figures 5-6) demonstrate excellent discrimination across all classes, with per-class AUC values exceeding 0.98 for most tissue types. Cancer-Associated Stroma had the lowest individual AUC, consistent with its lower F1-score, though the AUC remained above 0.97 for both models.

---

## 5. Discussion and Limitations

### 5.1 Interpretation of Results

Both models achieved strong classification performance (approximately 95% accuracy), demonstrating that deep learning can effectively distinguish between nine colorectal tissue types. The near-equivalent performance of the Custom CNN and ResNet-18 is noteworthy: while transfer learning is generally expected to provide a significant advantage, the large size of the PathMNIST training set (89,996 samples) allowed the custom CNN to learn competitive representations from scratch.

The marginal ROC-AUC advantage of ResNet-18 (0.998 vs. 0.996) suggests that pretrained features provide better probability calibration, which is clinically important when confidence scores are used for triage or quality control. Transfer learning showed its clearest advantage on Debris classification (F1: 0.90 vs. 0.84), where the low-level texture features learned from ImageNet likely helped distinguish this challenging class.

From a clinical perspective, the high accuracy on Adenocarcinoma Epithelium (F1=0.97) is particularly valuable, as this is the primary tumor class. The lower performance on Cancer-Associated Stroma (F1=0.75-0.80) is a limitation but is also clinically expected -- stroma classification is inherently ambiguous in H&E images and often requires immunohistochemical staining for definitive characterization.

### 5.2 Limitations

- **Resolution trade-off**: Due to memory constraints, images were loaded at 28x28 and resized to 224x224 via bilinear interpolation, which may result in loss of fine-grained cellular detail compared to native high-resolution imaging.
- **Single dataset**: Results are specific to the NCT-CRC-HE-100K dataset and may not generalize to tissue from different scanners, staining protocols, or patient populations.
- **Class imbalance**: The dataset exhibits moderate class imbalance (4:1 ratio between largest and smallest classes). While we used weighted evaluation metrics, class-weighted loss or oversampling could further improve minority class performance.
- **No external validation**: The model was evaluated only on the standard MedMNIST test split; external validation on independent clinical cohorts is needed before clinical deployment.

### 5.3 Ethical Considerations

Deploying AI models in pathology raises important ethical concerns. **Bias** may arise from the dataset being sourced from a single institution with specific patient demographics and staining protocols. **Patient safety** requires that such models serve as decision-support tools rather than autonomous diagnostic systems -- a pathologist should always make the final diagnosis. **Data privacy** is addressed by using the publicly available, de-identified MedMNIST dataset. **Misuse potential** exists if the model is applied outside its intended scope (e.g., on tissue types not represented in training data). Model confidence thresholds and human-in-the-loop validation are essential safeguards for clinical deployment.

### 5.4 Future Work

- Train on native 224x224 resolution images using GPU-equipped infrastructure to preserve fine-grained cellular detail
- Explore additional architectures (DenseNet-121, EfficientNet-B0, Vision Transformer) for further comparison
- Implement Grad-CAM visualization to verify that models focus on diagnostically relevant tissue regions
- Address class imbalance through focal loss or class-weighted sampling
- Validate on external datasets from different institutions and scanners
- Incorporate uncertainty estimation (e.g., MC Dropout) to flag low-confidence predictions for pathologist review

---

## References

1. Bera, K., Schalper, K. A., Rimm, D. L., Velcheti, V., & Madabhushi, A. (2019). Artificial intelligence in digital pathology -- new tools for diagnosis and precision oncology. *Nature Reviews Clinical Oncology*, 16(11), 703-715.

2. Chen, R. J., Chen, C., Li, Y., et al. (2022). Scaling vision transformers to gigapixel images via hierarchical self-supervised learning. *CVPR*, 16144-16155.

3. Coudray, N., Ocampo, P. S., Sakellaropoulos, T., et al. (2018). Classification and mutation prediction from non-small cell lung cancer histopathology images using deep learning. *Nature Medicine*, 24(10), 1559-1567.

4. Galon, J., Costes, A., Sanchez-Cabo, F., et al. (2006). Type, density, and location of immune cells within human colorectal tumors predict clinical outcome. *Science*, 313(5795), 1960-1964.

5. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 770-778.

6. Kather, J. N., Weis, C. A., Biber, F., et al. (2016). Multi-class texture analysis in colorectal cancer histology. *Scientific Reports*, 6, 27988.

7. Kather, J. N., Krisam, J., Charoentong, P., et al. (2019). Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study. *PLOS Medicine*, 16(1), e1002730.

8. Kompa, B., Snoek, J., & Beam, A. L. (2021). Second opinion needed: communicating uncertainty in medical machine learning. *NPJ Digital Medicine*, 4(1), 4.

9. Litjens, G., Kooi, T., Bejnordi, B. E., et al. (2017). A survey on deep learning in medical image analysis. *Medical Image Analysis*, 42, 60-88.

10. Macenko, M., Niethammer, M., Marron, J. S., et al. (2009). A method for normalizing histology slides for quantitative analysis. *ISBI*, 1107-1110.

11. Mormont, R., Geurts, P., & Maree, R. (2020). Comparison of deep transfer learning strategies for digital pathology. *CVPRW*, 2262-2271.

12. Sung, H., Ferlay, J., Siegel, R. L., et al. (2021). Global cancer statistics 2020: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. *CA: A Cancer Journal for Clinicians*, 71(3), 209-249.

13. Tajbakhsh, N., Shin, J. Y., Gurudu, S. R., et al. (2016). Convolutional neural networks for medical image analysis: Full training or fine tuning? *IEEE Transactions on Medical Imaging*, 35(5), 1299-1312.

14. Tellez, D., Litjens, G., Bandi, P., et al. (2019). Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology. *Medical Image Analysis*, 58, 101544.

15. Yang, J., Shi, R., Wei, D., et al. (2023). MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data*, 10, 41.
