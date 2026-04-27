# 🧠 CIFAR-10 CNN Classifier — Streamlit Demo App

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit)](https://your-app.streamlit.app)
[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)

> An interactive, production-grade Streamlit web app for real-time CIFAR-10 image classification using a custom ResNet-inspired CNN with **88%+ test accuracy**.

---

## ✨ Features

| Page | Description |
|---|---|
| 🖼️ **Live Prediction** | Upload any image → instant top-5 predictions with confidence chart |
| 🔥 **GradCAM Explorer** | Visual explanation — see *where* the model looks with heatmap overlay |
| 📋 **Model Report** | Full evaluation: confusion matrix, per-class F1, training curves |
| 🔬 **Dataset Explorer** | CIFAR-10 EDA: class distribution, sample grid, augmentation preview |
| 📦 **Batch Inference** | Upload 50 images → download predictions as CSV |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Copy trained checkpoint from the training repo
mkdir -p checkpoints
cp ../checkpoints/best_model.pth ./checkpoints/best_model.pth

# 3. Run the app
streamlit run app.py
```

> **Without a checkpoint:** the app runs with random weights and shows a warning banner — all pages are still fully functional for demonstration.

---

## 📁 Project Structure

```
cifar10-streamlit-app/
├── app.py                      ← Home page + model loader
├── requirements.txt
├── .streamlit/config.toml      ← Dark theme
├── pages/
│   ├── 1_Live_Prediction.py
│   ├── 2_GradCAM_Explorer.py
│   ├── 3_Model_Report.py
│   ├── 4_Dataset_Explorer.py
│   └── 5_Batch_Inference.py
├── models/cnn_model.py         ← CNN architecture (synced with training repo)
├── utils/
│   ├── model_loader.py         ← Checkpoint loading with fallback
│   ├── preprocessing.py        ← Image → tensor pipeline
│   ├── gradcam.py              ← GradCAM with hook system
│   └── visualization.py       ← Plotly + matplotlib helpers
└── checkpoints/                ← Place best_model.pth here
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push this folder to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Point to `streamlit_app/app.py` as the entry point.
4. Add `checkpoints/best_model.pth` via Streamlit Secrets or Git LFS.

---

## 🛠️ Technical Stack

- **Model:** Custom ResNet-CNN (~2.8M params), PyTorch 2.x
- **Interpretability:** GradCAM (gradient hooks, no model surgery)
- **Visualizations:** Plotly (interactive), Matplotlib/Seaborn (static)
- **Caching:** `@st.cache_resource` for model, `@st.cache_data` for datasets
- **Theme:** Custom dark theme via `.streamlit/config.toml`

---

## 👤 Author

**Your Name**
- 💼 [LinkedIn](https://linkedin.com/in/yourprofile)
- 🐙 [GitHub](https://github.com/yourusername)

---

## 📄 License

MIT License — see [LICENSE](../LICENSE).
