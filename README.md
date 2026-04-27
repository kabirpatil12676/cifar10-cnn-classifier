# ðŸ§  CIFAR-10 Image Classification with Custom ResNet-CNN

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-90.22%25-brightgreen)]()
[![Stars](https://img.shields.io/github/stars/kabirpatil12676/cifar10-cnn-classifier?style=social)](https://github.com/kabirpatil12676/cifar10-cnn-classifier)

> A production-grade deep learning pipeline for image classification on CIFAR-10, featuring a custom ResNet-inspired CNN with residual skip connections, label-smoothing loss, mixed-precision training, GradCAM interpretability, and a full evaluation + visualization suite.

---

## ðŸ–¼ï¸ Demo Visualizations

### Training Curves
![Training Curves](assets/training_curves.png?v=2)

### Confusion Matrix
![Confusion Matrix](assets/confusion_matrix.png?v=2)

### Per-Class Accuracy
![Per-Class Accuracy](assets/per_class_accuracy.png?v=2)

### GradCAM Interpretability
![GradCAM Example](assets/gradcam_example.png?v=2)

### Sample Images (CIFAR-10 Dataset)
![Sample Grid](assets/sample_grid.png?v=2)

---

## ðŸ“Š Results

| Class        | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| Airplane     | 0.91      | 0.90   | 0.90     |
| Automobile   | 0.93      | 0.95   | 0.94     |
| Bird         | 0.84      | 0.82   | 0.83     |
| Cat          | 0.79      | 0.77   | 0.78     |
| Deer         | 0.88      | 0.90   | 0.89     |
| Dog          | 0.82      | 0.83   | 0.82     |
| Frog         | 0.92      | 0.93   | 0.92     |
| Horse        | 0.93      | 0.92   | 0.92     |
| Ship         | 0.93      | 0.94   | 0.93     |
| Truck        | 0.92      | 0.93   | 0.93     |
| **Macro Avg**| **0.89**  | **0.89**| **0.89**|

**Overall Test Accuracy: 88.7%**

---

## ðŸ—ï¸ Architecture

```
Input (3Ã—32Ã—32)
     â”‚
     â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Block 1: ResidualBlock (3 â†’ 64)        â”‚
â”‚  Conv2d(3,64,3,p=1) â†’ BN â†’ ReLU        â”‚
â”‚  Conv2d(64,64,3,p=1) â†’ BN              â”‚
â”‚  Skip: Conv1Ã—1(3â†’64) â†’ BN              â”‚
â”‚  + Add â†’ ReLU â†’ MaxPool2d(2,2)         â”‚
â”‚  Output: (64 Ã— 16 Ã— 16)               â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
     â”‚
     â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Block 2: ResidualBlock (64 â†’ 128)      â”‚
â”‚  Conv2d(64,128,3,p=1) â†’ BN â†’ ReLU      â”‚
â”‚  Conv2d(128,128,3,p=1) â†’ BN            â”‚
â”‚  Skip: Conv1Ã—1(64â†’128) â†’ BN            â”‚
â”‚  + Add â†’ ReLU â†’ MaxPool2d(2,2)         â”‚
â”‚  Output: (128 Ã— 8 Ã— 8)                â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
     â”‚
     â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Block 3: ResidualBlock (128 â†’ 256)     â”‚
â”‚  Conv2d(128,256,3,p=1) â†’ BN â†’ ReLU     â”‚
â”‚  Conv2d(256,256,3,p=1) â†’ BN            â”‚
â”‚  Skip: Conv1Ã—1(128â†’256) â†’ BN           â”‚
â”‚  + Add â†’ ReLU â†’ AdaptiveAvgPool2d(2,2) â”‚
â”‚  Output: (256 Ã— 2 Ã— 2)                â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
     â”‚
     â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  Head: Classification                   â”‚
â”‚  Flatten â†’ Linear(1024â†’512) â†’ ReLU     â”‚
â”‚  Dropout(0.4) â†’ Linear(512â†’10)         â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
     â”‚
     â–¼
 Logits (10)
```

**Total Parameters: ~2.8M** &nbsp;|&nbsp; Trained for 38 epochs on CIFAR-10 (50K train / 10K test)

---

## âš¡ Quick Start

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

## ðŸ“ Project Structure

```
cifar10-cnn-classifier/
â”œâ”€â”€ README.md
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ setup.py
â”œâ”€â”€ assets/                      â† README visualizations
â”œâ”€â”€ config/
â”‚   â””â”€â”€ config.yaml              â† All hyperparameters in one place
â”œâ”€â”€ data/
â”‚   â””â”€â”€ dataloader.py            â† Dataset, augmentation, splits
â”œâ”€â”€ models/
â”‚   â””â”€â”€ cnn_model.py             â† ResNet-inspired CNN architecture
â”œâ”€â”€ training/
â”‚   â”œâ”€â”€ trainer.py               â† Training loop, early stopping, AMP
â”‚   â””â”€â”€ losses.py                â† Label Smoothing CrossEntropy
â”œâ”€â”€ evaluation/
â”‚   â””â”€â”€ evaluator.py             â† Accuracy, F1, confusion matrix
â”œâ”€â”€ visualization/
â”‚   â”œâ”€â”€ plot_results.py          â† Loss curves, confusion matrix, charts
â”‚   â””â”€â”€ gradcam.py               â† GradCAM interpretability
â”œâ”€â”€ utils/
â”‚   â”œâ”€â”€ logger.py                â† Structured timestamped logging
â”‚   â””â”€â”€ seed.py                  â† Full reproducibility seeding
â”œâ”€â”€ main.py                      â† CLI: train / eval / visualize
â”œâ”€â”€ inference.py                 â† Single-image prediction
â”œâ”€â”€ streamlit_app/               â† Interactive web demo
â””â”€â”€ notebooks/
    â””â”€â”€ CIFAR10_Analysis.ipynb
```

---

## ðŸ”‘ Key Techniques

- **Residual Skip Connections** â€” Prevents vanishing gradients, enables deeper learning
- **Batch Normalization** â€” Stabilizes training, reduces internal covariate shift
- **Data Augmentation** â€” RandomCrop, HorizontalFlip, ColorJitter to prevent overfitting
- **Label Smoothing Loss** â€” Prevents overconfident predictions (smoothing=0.1)
- **AdamW + CosineAnnealingLR** â€” Better convergence than fixed LR schedules
- **Mixed Precision Training** â€” 2Ã— faster training on supported GPUs
- **Early Stopping** â€” Prevents overfitting with patience=15
- **GradCAM** â€” Visual interpretability: see what the model focuses on
- **Proper Normalization** â€” CIFAR-10 channel-wise stats (not generic 0.5)
- **Reproducibility** â€” Fixed seeds across torch, numpy, random, cuda

---

## ðŸŒ Interactive Demo

A full multi-page Streamlit app is included in [`streamlit_app/`](streamlit_app/):

| Page | Description |
|---|---|
| ðŸ–¼ï¸ Live Prediction | Upload any image â†’ top-5 predictions with confidence chart |
| ðŸ”¥ GradCAM Explorer | Visualise where the model looks with heatmap overlay |
| ðŸ“‹ Model Report | Full evaluation: confusion matrix, per-class F1, training curves |
| ðŸ”¬ Dataset Explorer | CIFAR-10 EDA: class distribution, sample grid, augmentation preview |
| ðŸ“¦ Batch Inference | Upload 50 images â†’ download predictions as CSV |

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

---

## ðŸ› ï¸ Configuration

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

## ðŸ“‹ Requirements

- Python 3.9+
- PyTorch 2.3+
- CUDA 11.8+ (optional, for GPU training)
- See `requirements.txt` for full list

---

## ðŸ‘¤ Author

**Kabir Patil**
- ðŸ”— [LinkedIn](https://www.linkedin.com/in/kabir-patil-7a2a9b30b/)
- ðŸ™ [GitHub](https://github.com/kabirpatil12676)

---

## ðŸ“„ License

This project is licensed under the MIT License â€” see [LICENSE](LICENSE) for details.

