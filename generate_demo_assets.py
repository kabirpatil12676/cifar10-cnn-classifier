"""
generate_demo_assets.py
========================
Generates README visualizations using REAL CIFAR-10 data and the actual
training history from checkpoints/training_history.json.

Run:  python generate_demo_assets.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent
ASSETS_DIR   = ROOT / "assets"
HISTORY_FILE = ROOT / "checkpoints" / "training_history.json"
ASSETS_DIR.mkdir(exist_ok=True)

CLASSES = ["airplane","automobile","bird","cat","deer",
           "dog","frog","horse","ship","truck"]

# ── consistent dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0E1117",
    "axes.facecolor":   "#1A1D27",
    "axes.edgecolor":   "#374151",
    "axes.labelcolor":  "#E5E7EB",
    "xtick.color":      "#9CA3AF",
    "ytick.color":      "#9CA3AF",
    "text.color":       "#E5E7EB",
    "grid.color":       "#2D3748",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
})

# =============================================================================
# 1. REAL CIFAR-10 SAMPLE GRID
# =============================================================================
print("Downloading CIFAR-10 and extracting real images ...")
from torchvision.datasets import CIFAR10
from torchvision import transforms

data_root = str(ROOT / "data" / "cifar10_raw")
dataset   = CIFAR10(root=data_root, train=True, download=True,
                    transform=transforms.ToTensor())

# Collect 5 real images per class (first occurrence in the dataset)
N_PER_CLASS = 5
class_images = {i: [] for i in range(10)}
for img_tensor, label in dataset:
    if len(class_images[label]) < N_PER_CLASS:
        # Convert CHW float32 -> HWC uint8
        img_np = (img_tensor.permute(1,2,0).numpy() * 255).astype("uint8")
        class_images[label].append(img_np)
    if all(len(v) == N_PER_CLASS for v in class_images.values()):
        break

fig, axes = plt.subplots(10, N_PER_CLASS, figsize=(N_PER_CLASS * 1.6, 10 * 1.6))
fig.patch.set_facecolor("#0E1117")
fig.suptitle("CIFAR-10 Dataset — Real Sample Images (5 per class)",
             color="#F9FAFB", fontsize=14, y=1.01, fontweight="bold")

for row in range(10):
    for col in range(N_PER_CLASS):
        ax = axes[row, col]
        ax.imshow(class_images[row][col], interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#4F8EF7"); spine.set_linewidth(0.8)
        if col == 0:
            ax.set_ylabel(CLASSES[row].capitalize(), color="#D1D5DB",
                          fontsize=10, rotation=0, labelpad=60, va="center",
                          fontweight="bold")

plt.tight_layout(pad=0.4)
out = ASSETS_DIR / "sample_grid.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0E1117")
plt.close()
print(f"  Saved: {out}")

# =============================================================================
# 2. REAL TRAINING CURVES (from training_history.json)
# =============================================================================
print("Generating training curves from real history ...")

with open(HISTORY_FILE, encoding="utf-8") as f:
    history = json.load(f)

train_loss = history["train_loss"]
val_loss   = history["val_loss"]
train_acc  = history["train_acc"]
val_acc    = history["val_acc"]
epochs     = list(range(1, len(train_loss) + 1))

best_epoch = int(np.argmax(val_acc)) + 1
best_acc   = max(val_acc)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
fig.patch.set_facecolor("#0E1117")

# Loss
ax1.plot(epochs, train_loss, color="#F87171", lw=2,            label="Train Loss")
ax1.plot(epochs, val_loss,   color="#FBBF24", lw=2, ls="--",   label="Val Loss")
ax1.axvline(best_epoch, color="#4F8EF7", lw=1.2, ls=":",
            alpha=0.8, label=f"Best epoch ({best_epoch})")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title("Training & Validation Loss", fontsize=13, pad=10, color="#F9FAFB")
ax1.legend(facecolor="#1A1D27", edgecolor="#374151")
ax1.grid(True, alpha=0.4)

# Accuracy
ax2.plot(epochs, train_acc, color="#4F8EF7", lw=2,            label="Train Acc")
ax2.plot(epochs, val_acc,   color="#34D399", lw=2, ls="--",   label="Val Acc")
ax2.axhline(best_acc, color="#A78BFA", lw=1.2, ls=":",
            alpha=0.8, label=f"Best val acc ({best_acc:.1f}%)")
ax2.axvline(best_epoch, color="#4F8EF7", lw=1.2, ls=":", alpha=0.8)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Training & Validation Accuracy", fontsize=13, pad=10, color="#F9FAFB")
ax2.legend(facecolor="#1A1D27", edgecolor="#374151")
ax2.grid(True, alpha=0.4)

plt.tight_layout(pad=1.5)
out = ASSETS_DIR / "training_curves.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0E1117")
plt.close()
print(f"  Saved: {out}  (best epoch={best_epoch}, best_val_acc={best_acc:.2f}%)")

# =============================================================================
# 3. REAL CONFUSION MATRIX (from trained model on test set)
# =============================================================================
print("Running model on test set to get real confusion matrix ...")
import torch
sys.path.insert(0, str(ROOT))
from models.cnn_model import CIFAR10Net

device = torch.device("cpu")
model  = CIFAR10Net(num_classes=10, dropout_rate=0.4).to(device)
ckpt   = torch.load(ROOT / "checkpoints" / "best_model.pth",
                    map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)
test_tf = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=MEAN, std=STD)])
testset    = CIFAR10(root=data_root, train=False, download=True, transform=test_tf)
testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                         shuffle=False, num_workers=0)

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, lbls in testloader:
        preds = model(imgs.to(device)).argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(lbls.tolist())

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm      = confusion_matrix(all_labels, all_preds)
acc     = accuracy_score(all_labels, all_preds) * 100
report  = classification_report(all_labels, all_preds, target_names=CLASSES,
                                 output_dict=True)
print(f"  Real test accuracy: {acc:.2f}%")

# Confusion matrix plot
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cmap    = LinearSegmentedColormap.from_list(
    "cf", ["#0E1117","#1e3a5f","#2563EB","#60A5FA","#BFDBFE"])

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor("#0E1117"); ax.set_facecolor("#1A1D27")
sns.heatmap(cm_norm, annot=True, fmt=".2f",
            xticklabels=CLASSES, yticklabels=CLASSES,
            cmap=cmap, linewidths=0.4, linecolor="#2D3748",
            ax=ax, annot_kws={"size": 8.5},
            cbar_kws={"shrink": 0.8})
ax.set_title(f"Normalised Confusion Matrix — Test Set (Acc: {acc:.1f}%)",
             color="#F9FAFB", fontsize=13, pad=14)
ax.set_xlabel("Predicted Label", color="#9CA3AF")
ax.set_ylabel("True Label",      color="#9CA3AF")
ax.tick_params(colors="#9CA3AF", rotation=45, labelsize=9)
ax.collections[0].colorbar.ax.tick_params(colors="#9CA3AF")
plt.tight_layout()
out = ASSETS_DIR / "confusion_matrix.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0E1117")
plt.close()
print(f"  Saved: {out}")

# =============================================================================
# 4. REAL PER-CLASS ACCURACY
# =============================================================================
print("Generating per-class accuracy chart ...")
per_class_acc = 100.0 * cm.diagonal() / cm.sum(axis=1)
order         = np.argsort(per_class_acc)[::-1]
sorted_cls    = [CLASSES[i] for i in order]
sorted_acc    = per_class_acc[order]
colors        = ["#34D399" if a >= 90 else "#FBBF24" if a >= 85 else "#F87171"
                 for a in sorted_acc]

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#0E1117")
bars = ax.barh(sorted_cls, sorted_acc, color=colors, edgecolor="#1A1D27", height=0.65)
for bar, val in zip(bars, sorted_acc):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", ha="left", fontsize=9, color="#E5E7EB")
ax.axvline(per_class_acc.mean(), color="#A78BFA", lw=1.5, ls="--",
           label=f"Mean: {per_class_acc.mean():.1f}%")
ax.set_xlim(60, 108); ax.invert_yaxis()
ax.set_xlabel("Test Accuracy (%)"); ax.grid(True, axis="x", alpha=0.4)
ax.set_axisbelow(True)
ax.set_title(f"Per-Class Test Accuracy  (Overall: {acc:.1f}%)",
             fontsize=13, pad=10, color="#F9FAFB")
patches = [mpatches.Patch(color="#34D399", label=">=90%"),
           mpatches.Patch(color="#FBBF24", label="85-90%"),
           mpatches.Patch(color="#F87171", label="<85%")]
ax.legend(handles=patches, facecolor="#1A1D27", edgecolor="#374151",
          loc="lower right", fontsize=9)
plt.tight_layout()
out = ASSETS_DIR / "per_class_accuracy.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0E1117")
plt.close()
print(f"  Saved: {out}")

# =============================================================================
# 5. REAL GRADCAM on actual test images
# =============================================================================
print("Generating GradCAM on real test images ...")
import cv2

# Pick 3 test images: one easy (automobile), one medium (cat), one hard (dog)
target_classes = {"automobile": None, "cat": None, "dog": None}
for img_pil, lbl in CIFAR10(root=data_root, train=False, download=False):
    cls_name = CLASSES[lbl]
    if cls_name in target_classes and target_classes[cls_name] is None:
        target_classes[cls_name] = img_pil
    if all(v is not None for v in target_classes.values()):
        break

sys.path.insert(0, str(ROOT / "streamlit_app"))
from utils.gradcam import GradCAM

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
fig.patch.set_facecolor("#0E1117")
fig.suptitle("GradCAM Visualizations — Real Test Images\n(Block 3, 256ch → predicted class)",
             color="#F9FAFB", fontsize=13, y=1.02, fontweight="bold")

col_titles = ["Original Image", "GradCAM Heatmap", "Overlay (a=0.5)"]
for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, color="#D1D5DB", fontsize=11, pad=10)

for row, (cls_name, pil_img) in enumerate(target_classes.items()):
    # Preprocess
    img_rgb  = pil_img.convert("RGB")
    img_disp = np.array(img_rgb.resize((224, 224)))   # for display
    tf       = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(),
                                   transforms.Normalize(mean=MEAN, std=STD)])
    tensor   = tf(img_rgb).unsqueeze(0)

    # GradCAM
    gradcam  = GradCAM(model, model.block3)
    try:
        heatmap = gradcam.generate(tensor, device=device)
    finally:
        gradcam.remove_hooks()

    heatmap_big = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
    colored     = cv2.applyColorMap((heatmap_big * 255).astype("uint8"), cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay     = np.clip(0.5*colored_rgb.astype(float) + 0.5*img_disp.astype(float),
                          0, 255).astype("uint8")

    # Prediction
    with torch.no_grad():
        probs    = torch.softmax(model(tensor), dim=1).squeeze(0).cpu().numpy()
        pred_cls = CLASSES[int(np.argmax(probs))]
        conf     = float(probs.max())

    row_label = f"True: {cls_name}  |  Pred: {pred_cls} ({conf:.0%})"
    axes[row, 0].set_ylabel(row_label, color="#D1D5DB", fontsize=9,
                             rotation=0, labelpad=150, va="center")

    for col, img in enumerate([img_disp, colored_rgb, overlay]):
        ax = axes[row, col]
        ax.imshow(img, interpolation="bilinear")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#374151"); spine.set_linewidth(0.6)

plt.tight_layout(pad=1.5)
out = ASSETS_DIR / "gradcam_example.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0E1117")
plt.close()
print(f"  Saved: {out}")

# =============================================================================
# Summary
# =============================================================================
print()
print("=" * 55)
print(f"  Real test accuracy : {acc:.2f}%")
print(f"  Best val accuracy  : {best_acc:.2f}%  (epoch {best_epoch})")
print()
for cls in CLASSES:
    r = report[cls]
    print(f"  {cls:<12}  P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}")
print("=" * 55)
print("All real assets saved to assets/")
