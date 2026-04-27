"""
inference.py
=============
Single-image inference with confidence scores.

Loads a trained checkpoint and classifies any image file,
printing top-5 class probabilities and optionally displaying
a matplotlib confidence bar chart.

Usage:
    python inference.py --image path/to/image.jpg
    python inference.py --image photo.png --checkpoint checkpoints/best_model.pth
    python inference.py --image photo.png --no-plot
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from models.cnn_model import build_model
from utils.logger import get_logger
from utils.seed import set_seed

logger = get_logger(__name__)

# CIFAR-10 channel-wise stats
_MEAN = [0.4914, 0.4822, 0.4465]
_STD  = [0.2470, 0.2435, 0.2616]

_PREPROCESS = transforms.Compose([
    transforms.Resize((32, 32)),          # CIFAR-10 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_path: str, config: dict,
                    device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to ``.pth`` file.
        config:          Config dict.
        device:          Target device.

    Returns:
        Model in eval mode.
    """
    if not os.path.exists(checkpoint_path):
        logger.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)

    model = build_model(config)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    val_acc = ckpt.get("val_acc", float("nan"))
    epoch   = ckpt.get("epoch", "?")
    logger.info("Checkpoint loaded | epoch=%s | val_acc=%.2f%%", epoch, val_acc)
    return model


def predict(
    image_path: str,
    model: torch.nn.Module,
    class_names: list[str],
    device: torch.device,
    top_k: int = 5,
) -> list[dict]:
    """Run inference on a single image file.

    Args:
        image_path:  Path to any PIL-readable image (JPEG, PNG, …).
        model:       Loaded model in eval mode.
        class_names: List of 10 CIFAR-10 class label strings.
        device:      Torch device.
        top_k:       Number of top predictions to return.

    Returns:
        List of ``{"class": str, "confidence": float}`` dicts,
        sorted descending by confidence, length ``top_k``.

    Raises:
        FileNotFoundError: If ``image_path`` does not exist.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img   = Image.open(image_path).convert("RGB")
    tensor = _PREPROCESS(img).unsqueeze(0).to(device)  # (1, 3, 32, 32)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_indices = np.argsort(probs)[::-1][:top_k]
    return [
        {"class": class_names[i], "confidence": float(probs[i])}
        for i in top_indices
    ]


def display_results(
    image_path: str,
    predictions: list[dict],
    show_plot: bool = True,
    save_path: str | None = None,
) -> None:
    """Print and optionally plot top-K predictions.

    Args:
        image_path:  Path to original image.
        predictions: Output from :func:`predict`.
        show_plot:   Whether to display the matplotlib figure.
        save_path:   If given, save the figure here instead of showing it.
    """
    # Terminal output
    print("\n" + "=" * 45)
    print(f"  Image : {os.path.basename(image_path)}")
    print(f"  Top-{len(predictions)} Predictions")
    print("=" * 45)
    for rank, pred in enumerate(predictions, 1):
        bar  = "█" * int(pred["confidence"] * 30)
        print(f"  {rank}. {pred['class']:<12} {pred['confidence']:6.1%}  {bar}")
    print("=" * 45)

    if not show_plot and save_path is None:
        return

    # Matplotlib figure
    img_rgb = Image.open(image_path).convert("RGB").resize((128, 128))
    classes = [p["class"] for p in predictions]
    confs   = [p["confidence"] for p in predictions]
    colors  = ["#2ECC71" if i == 0 else "#3498DB" for i in range(len(predictions))]

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("CIFAR-10 Inference", fontsize=13, fontweight="bold")

    ax_img.imshow(img_rgb)
    ax_img.axis("off")
    ax_img.set_title(
        f"Predicted: {predictions[0]['class'].upper()}\n"
        f"Confidence: {predictions[0]['confidence']:.1%}",
        fontsize=11, color="#2ECC71", fontweight="bold",
    )

    bars = ax_bar.barh(classes[::-1], confs[::-1], color=colors[::-1],
                       edgecolor="white", height=0.5)
    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_xlabel("Confidence")
    ax_bar.set_title(f"Top-{len(predictions)} Class Probabilities")
    ax_bar.grid(axis="x", alpha=0.3)
    for bar, conf in zip(bars, confs[::-1]):
        ax_bar.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{conf:.1%}", va="center", fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Inference plot saved → %s", save_path)
    if show_plot:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CIFAR-10 single-image inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python inference.py --image photo.jpg\n"
            "  python inference.py --image photo.jpg --no-plot\n"
            "  python inference.py --image photo.jpg --save results/pred.png\n"
        ),
    )
    parser.add_argument("--image",      type=str, required=True,
                        help="Path to input image file.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pth checkpoint (overrides config).")
    parser.add_argument("--config",     type=str, default="config/config.yaml",
                        help="Config YAML path (default: config/config.yaml).")
    parser.add_argument("--top-k",      type=int, default=5,
                        help="Number of top predictions to show (default: 5).")
    parser.add_argument("--no-plot",    action="store_true",
                        help="Suppress the matplotlib figure.")
    parser.add_argument("--save",       type=str, default=None,
                        help="Save inference plot to this path.")
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    ckpt_path   = args.checkpoint or config["evaluation"]["checkpoint_path"]
    class_names = list(config["classes"])

    model       = load_checkpoint(ckpt_path, config, device)
    predictions = predict(args.image, model, class_names, device, top_k=args.top_k)

    display_results(
        image_path=args.image,
        predictions=predictions,
        show_plot=not args.no_plot,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
