"""
utils/preprocessing.py — AUDIT FIXED
FIXES: Image.Resampling.NEAREST (Pillow 10+), normalized preview clamp,
       tensor.clone().detach() in denormalize, removed unused _DISPLAY_TRANSFORM.
"""
from __future__ import annotations
import io
from typing import Tuple
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD:  Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck",
]
_INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
])
# FIX: Pillow 10+ deprecated Image.NEAREST as top-level constant
_NEAREST = Image.Resampling.NEAREST

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """PIL image → normalised (1,3,32,32) tensor."""
    return _INFERENCE_TRANSFORM(image.convert("RGB")).unsqueeze(0)

def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """Reverse CIFAR-10 normalisation → HWC uint8 numpy array."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    mean = torch.tensor(CIFAR10_MEAN).view(3,1,1)
    std  = torch.tensor(CIFAR10_STD).view(3,1,1)
    img  = tensor.clone().detach() * std + mean
    img  = img.permute(1,2,0).cpu().numpy()
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

def pil_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """PIL image → raw bytes."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()

def get_augmentation_samples(image: Image.Image) -> dict[str, Image.Image]:
    """Six augmented 128×128 PIL images for the preview grid."""
    img32  = image.convert("RGB").resize((32, 32))
    to_pil = transforms.ToPILImage()
    aug    = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    ])
    # FIX: clamp normalised tensor back to [0,1] before ToPILImage
    norm_t = _INFERENCE_TRANSFORM(img32)
    norm_pil = to_pil(torch.clamp(
        norm_t * torch.tensor(CIFAR10_STD).view(3,1,1)
              + torch.tensor(CIFAR10_MEAN).view(3,1,1), 0.0, 1.0
    ))
    samples = {
        "Original":     img32,
        "H-Flip":       transforms.functional.hflip(img32),
        "Random Crop":  transforms.RandomCrop(32, padding=4)(img32),
        "Color Jitter": transforms.ColorJitter(brightness=0.4, contrast=0.4)(img32),
        "All Combined": aug(img32),
        "Normalized":   norm_pil,
    }
    return {k: v.resize((128, 128), _NEAREST) for k, v in samples.items()}
