#!/usr/bin/env python3
"""Classify an image as trash or recycling."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from label_mapping import CLASS_NAMES
from model_def import build_resnet18_binary
from util_device import get_torch_device


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_resnet18_binary(num_classes=2, weights=None)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device), ckpt


def build_eval_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


def classify_pil(
    model: torch.nn.Module,
    pil_image: Image.Image,
    device: torch.device,
    img_size: int,
) -> tuple[str, float, list[float]]:
    """Run inference on an RGB PIL image (e.g. from camera or upload)."""
    tf = build_eval_transform(img_size)
    x = tf(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())
    return CLASS_NAMES[idx], float(probs[idx]), [float(p) for p in probs]


def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    img_size: int,
) -> tuple[str, float, list[float]]:
    img = Image.open(image_path).convert("RGB")
    return classify_pil(model, img, device, img_size)


def main() -> None:
    ap = argparse.ArgumentParser(description="Trash vs recycling inference.")
    ap.add_argument("image", type=Path, help="Path to an image file")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints" / "model.pt",
    )
    args = ap.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\nTrain first: python train.py --data-root <images_dir>"
        )

    device = get_torch_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    img_size = int(ckpt.get("img_size", 224))

    label, conf, probs = predict_image(model, args.image, device, img_size)
    print(f"Prediction: {label} ({conf:.1%} confidence)")
    for name, p in zip(CLASS_NAMES, probs):
        print(f"  {name}: {p:.1%}")


if __name__ == "__main__":
    main()
