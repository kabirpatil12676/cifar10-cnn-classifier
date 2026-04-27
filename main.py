"""
main.py
========
CLI entry point for the CIFAR-10 CNN Classifier.

Supported modes:
    train     — Download data, train model, save best checkpoint.
    eval      — Load checkpoint, run full evaluation, print metrics.
    visualize — Load checkpoint + history, generate all result plots.

Usage examples:
    python main.py --mode train
    python main.py --mode eval   --checkpoint checkpoints/best_model.pth
    python main.py --mode visualize --checkpoint checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import yaml
import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar/array types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from data.dataloader import get_dataloaders, get_class_names
from evaluation.evaluator import Evaluator, load_model_for_eval
from models.cnn_model import build_model
from training.trainer import Trainer
from utils.logger import get_logger
from utils.seed import set_seed
from visualization.gradcam import GradCAM
from visualization.plot_results import ResultPlotter

logger = get_logger(__name__, log_dir="logs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load and return the YAML configuration dictionary."""
    if not os.path.exists(config_path):
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s | VRAM: %.1f GB",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)
    return device


def save_history(history_dict: dict, path: str) -> None:
    """Persist training history to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2)
    logger.info("Training history saved → %s", path)


def load_history(path: str) -> dict:
    """Load training history from a JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"History file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def run_train(config: dict) -> None:
    """Download data, train model, checkpoint best weights."""
    device = get_device()
    set_seed(config["seed"])

    logger.info("=" * 60)
    logger.info("MODE: TRAIN")
    logger.info("=" * 60)

    train_loader, val_loader, _ = get_dataloaders(config)
    logger.info("Dataset: %d train | %d val samples",
                len(train_loader.dataset), len(val_loader.dataset))

    model   = build_model(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s | Parameters: %s",
                config["model"]["name"], f"{n_params:,}")

    trainer = Trainer(model, config, train_loader, val_loader, device)
    history = trainer.train()

    history_path = os.path.join(
        config["training"]["checkpoint_dir"], "training_history.json"
    )
    save_history(history.as_dict(), history_path)
    logger.info("Training complete. Run --mode eval to evaluate.")


def run_eval(config: dict, checkpoint_path: str) -> dict:
    """Load checkpoint and run full evaluation on the test set."""
    device = get_device()
    set_seed(config["seed"])

    logger.info("=" * 60)
    logger.info("MODE: EVALUATE")
    logger.info("=" * 60)

    _, _, test_loader = get_dataloaders(config)
    class_names       = get_class_names(config)

    model = build_model(config)
    model = load_model_for_eval(checkpoint_path, model, device)

    evaluator = Evaluator(model, test_loader, class_names, device)
    results   = evaluator.evaluate()

    print("\n" + "=" * 60)
    print(f"  Test Accuracy : {results['accuracy']:.2f}%")
    print(f"  Macro F1      : {results['macro_f1']:.4f}")
    print(f"  Weighted F1   : {results['weighted_f1']:.4f}")
    print("=" * 60)
    print("\nPer-class breakdown:")
    print(results["report"])

    # Save metrics JSON
    metrics_path = os.path.join(
        config["visualization"]["results_dir"], "metrics.json"
    )
    os.makedirs(config["visualization"]["results_dir"], exist_ok=True)
    serialisable = {
        k: v for k, v in results.items()
        if k not in ("confusion_matrix", "worst_samples", "all_probs",
                     "all_preds", "all_labels")
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, cls=_NumpyEncoder)
    logger.info("Metrics saved → %s", metrics_path)

    return results


def run_visualize(config: dict, checkpoint_path: str) -> None:
    """Generate all plots and save to results/ directory."""
    device = get_device()
    set_seed(config["seed"])

    logger.info("=" * 60)
    logger.info("MODE: VISUALIZE")
    logger.info("=" * 60)

    train_loader, _, test_loader = get_dataloaders(config)
    class_names = get_class_names(config)

    model = build_model(config)
    model = load_model_for_eval(checkpoint_path, model, device)

    # Run evaluation to get confusion matrix and probs
    evaluator = Evaluator(model, test_loader, class_names, device)
    results   = evaluator.evaluate()

    plotter = ResultPlotter(config, class_names)

    # 1. Sample grid
    plotter.plot_sample_grid(train_loader)

    # 2. Training curves — load history if available
    history_path = os.path.join(
        config["training"]["checkpoint_dir"], "training_history.json"
    )
    if os.path.exists(history_path):
        history = load_history(history_path)
        plotter.plot_training_curves(history)
    else:
        logger.warning("Training history not found at %s — skipping curves.", history_path)

    # 3. Confusion matrix
    plotter.plot_confusion_matrix(results["confusion_matrix"])

    # 4. Per-class accuracy
    plotter.plot_per_class_accuracy(results["confusion_matrix"])

    # 5. Confidence histogram
    plotter.plot_confidence_histogram(
        results["all_probs"], results["all_preds"], results["all_labels"]
    )

    # 6. Worst samples
    if results["worst_samples"]:
        plotter.plot_worst_samples(results["worst_samples"])

    # 7. GradCAM
    target_layer = model.block3
    gradcam = GradCAM(model, target_layer)
    gradcam.generate_batch_plots(test_loader, class_names, config, device)
    gradcam.remove_hooks()

    logger.info("All plots saved to: %s", config["visualization"]["results_dir"])
    print(f"\n✓ All visualisations saved to '{config['visualization']['results_dir']}/'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CIFAR-10 CNN Classifier — train, evaluate, or visualize.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["train", "eval", "visualize"],
        help="Operation mode: train / eval / visualize",
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to YAML config file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint .pth (required for eval and visualize)",
    )
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)

    if args.mode == "train":
        run_train(config)

    elif args.mode == "eval":
        ckpt = args.checkpoint or config["evaluation"]["checkpoint_path"]
        run_eval(config, ckpt)

    elif args.mode == "visualize":
        ckpt = args.checkpoint or config["evaluation"]["checkpoint_path"]
        run_visualize(config, ckpt)


if __name__ == "__main__":
    main()
