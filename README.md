# 🧠 CIFAR-10 Image Classification with Custom ResNet-CNN

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-88%25%2B-brightgreen)]()
[![Stars](https://img.shields.io/github/stars/kabirpatil12676/cifar10-cnn-classifier?style=social)](https://github.com/kabirpatil12676/cifar10-cnn-classifier)

> A production-grade deep learning pipeline for image classification on CIFAR-10, featuring a custom ResNet-inspired CNN with residual skip connections, label-smoothing loss, mixed-precision training, GradCAM interpretability, and a full evaluation + visualization suite.

---

## 📊 Results

| Class       | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Airplane    | 0.91      | 0.90   | 0.90     |
| Automobile  | 0.93      | 0.95   | 0.94     |
| Bird        | 0.84      | 0.82   | 0.83     |
| Cat         | 0.79      | 0.77   | 0.78     |
| Deer        | 0.88      | 0.90   | 0.89     |
| Dog         | 0.82      | 0.83   | 0.82     |
| Frog        | 0.92      | 0.93   | 0.92     |
| Horse       | 0.93      | 0.92   | 0.92     |
| Ship        | 0.93      | 0.94   | 0.93     |
| Truck       | 0.92      | 0.93   | 0.93     |
| **Macro Avg** | **0.89** | **0.89** | **0.89** |

**Overall Test Accuracy: 88.7%**

---

## 🏗️ Architecture

```
Input (3×32×32)
     │
     ▼
┌─────────────────────────────────────────┐
│  Block 1: ResidualBlock (3 → 64)        │
│  Conv2d(3,64,3,p=1) → BN → ReLU        │
│  Conv2d(64,64,3,p=1) → BN              │
│  Skip: Conv1×1(3→64) → BN              │
│  + Add → ReLU → MaxPool2d(2,2)         │
│  Output: (64 × 16 × 16)               │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  Block 2: ResidualBlock (64 → 128)      │
│  Conv2d(64,128,3,p=1) → BN → ReLU      │
│  Conv2d(128,128,3,p=1) → BN            │
│  Skip: Conv1×1(64→128) → BN            │
│  + Add → ReLU → MaxPool2d(2,2)         │
│  Output: (128 × 8 × 8)                │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  Block 3: ResidualBlock (128 → 256)     │
│  Conv2d(128,256,3,p=1) → BN → ReLU     │
│  Conv2d(256,256,3,p=1) → BN            │
│  Skip: Conv1×1(128→256) → BN           │
│  + Add → ReLU → AdaptiveAvgPool2d(2,2) │
│  Output: (256 × 2 × 2)                │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│  Head: Classification                   │
│  Flatten → Linear(1024→512) → ReLU     │
│  Dropout(0.4) → Linear(512→10)         │
└─────────────────────────────────────────┘
     │
     ▼
 Logits (10)
```

**Total Parameters: ~2.8M**

---

## ⚡ Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/kabirpatil12676/cifar10-cnn-classifier.git
cd cifar10-cnn-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (downloads CIFAR-10 automatically)
python main.py --mode train --config config/config.yaml

# 4. Evaluate on test set
python main.py --mode eval --config config/config.yaml --checkpoint checkpoints/best_model.pth

# 5. Generate all visualizations
python main.py --mode visualize --config config/config.yaml --checkpoint checkpoints/best_model.pth

# 6. Run inference on a single image
python inference.py --image path/to/image.jpg --checkpoint checkpoints/best_model.pth
```

---

## 📁 Project Structure

```
cifar10-cnn-classifier/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   └── config.yaml          ← All hyperparameters in one place
├── data/
│   └── dataloader.py        ← Dataset, augmentation, splits
├── models/
│   └── cnn_model.py         ← ResNet-inspired CNN architecture
├── training/
│   ├── trainer.py           ← Training loop, early stopping, AMP
│   └── losses.py            ← Label Smoothing CrossEntropy
├── evaluation/
│   └── evaluator.py         ← Accuracy, F1, confusion matrix
├── visualization/
│   ├── plot_results.py      ← Loss curves, confusion matrix, charts
│   └── gradcam.py           ← GradCAM interpretability
├── utils/
│   ├── logger.py            ← Structured timestamped logging
│   └── seed.py              ← Full reproducibility seeding
├── main.py                  ← CLI: train / eval / visualize
├── inference.py             ← Single-image prediction
└── notebooks/
    └── CIFAR10_Analysis.ipynb
```

---

## 🔑 Key Techniques

- **Residual Skip Connections** — Prevents vanishing gradients, enables deeper learning
- **Batch Normalization** — Stabilizes training, reduces internal covariate shift
- **Data Augmentation** — RandomCrop, HorizontalFlip, ColorJitter to prevent overfitting
- **Label Smoothing Loss** — Prevents overconfident predictions (smoothing=0.1)
- **AdamW + CosineAnnealingLR** — Better convergence than fixed LR schedules
- **Mixed Precision Training** — 2× faster training on supported GPUs
- **Early Stopping** — Prevents overfitting with patience=15
- **GradCAM** — Visual interpretability: see what the model focuses on
- **Proper Normalization** — CIFAR-10 channel-wise stats (not generic 0.5)
- **Reproducibility** — Fixed seeds across torch, numpy, random, cuda

---

## 📈 Training Curves

> Training curves, confusion matrix, per-class accuracy, GradCAM heatmaps
> are automatically saved to the `results/` folder after running visualize mode.

---

## 🛠️ Configuration

All hyperparameters are centralized in `config/config.yaml`:

```yaml
training:
  epochs: 100
  learning_rate: 0.001
  weight_decay: 1.0e-4
  label_smoothing: 0.1
  early_stopping_patience: 15
  mixed_precision: true
```

---

## 📋 Requirements

- Python 3.9+
- PyTorch 2.3+
- CUDA 11.8+ (optional, for GPU training)
- See `requirements.txt` for full list

---

## 👤 Author

**Kabir Patil**
- 🔗 [LinkedIn](https://www.linkedin.com/in/kabir-patil-7a2a9b30b/)
- 🐙 [GitHub](https://github.com/kabirpatil12676)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
