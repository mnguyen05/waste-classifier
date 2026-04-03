#!/usr/bin/env python3
"""Train trash vs recycling classifier (ResNet-18, transfer learning)."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from dataset import WasteBinaryDataset, collect_samples
from label_mapping import CLASS_NAMES, CATEGORY_TO_LABEL
from model_def import build_resnet18_binary, load_pretrained_resnet18
from util_device import get_torch_device

IMG_SIZE = 224


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_split(
    labels: list[int], val_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {0: [], 1: []}
    for i, y in enumerate(labels):
        by_class[y].append(i)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for y in (0, 1):
        idxs = by_class[y]
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_ratio))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def resolve_data_root(cli: Path | None) -> Path:
    if cli is not None:
        return cli.expanduser().resolve()
    env = os.environ.get("WASTE_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    raise SystemExit(
        "Missing data path. Pass --data-root /path/to/images or set WASTE_DATA_ROOT."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train trash vs recycling model.")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Folder containing waste category subfolders (or parent with images/images/).",
    )
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints" / "model.pt",
    )
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument(
        "--from-scratch",
        action="store_true",
        help="Skip ImageNet weights (random init).",
    )
    args = ap.parse_args()

    data_root = resolve_data_root(args.data_root)
    device = get_torch_device()
    set_seed(args.seed)
    print(f"Device: {device}")
    print(f"Data: {data_root}")

    paths, labels = collect_samples(data_root)
    train_idx, val_idx = stratified_split(labels, args.val_ratio, args.seed)
    print(f"Samples: {len(paths)}  train: {len(train_idx)}  val: {len(val_idx)}")

    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(0.12, 0.12, 0.08, 0.04),
            transforms.ToTensor(),
            norm,
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            norm,
        ]
    )

    train_ds = Subset(WasteBinaryDataset(paths, labels, train_tf), train_idx)
    val_ds = Subset(WasteBinaryDataset(paths, labels, val_tf), val_idx)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    if args.from_scratch:
        print("Training from scratch (--from-scratch).")
        model = build_resnet18_binary(num_classes=2, weights=None).to(device)
    else:
        try:
            model = build_resnet18_binary(
                num_classes=2, weights=load_pretrained_resnet18()
            ).to(device)
            print("Using ImageNet-pretrained ResNet-18.")
        except Exception as e:
            print(f"Could not load pretrained weights ({e!s}). Training from scratch.")
            model = build_resnet18_binary(num_classes=2, weights=None).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs * steps_per_epoch)
    )
    crit = nn.CrossEntropyLoss()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{sum(losses) / len(losses):.4f}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_acc = correct / total if total else 0.0
        train_loss = sum(losses) / len(losses) if losses else 0.0
        print(
            f"Epoch {epoch:02d}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            payload = {
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "category_mapping": dict(CATEGORY_TO_LABEL),
                "img_size": IMG_SIZE,
                "backbone": "resnet18",
                "val_acc": val_acc,
                "epoch": epoch,
            }
            torch.save(payload, args.output)
            print(f"  → saved best checkpoint ({args.output})")

    meta = {
        "checkpoint": str(args.output),
        "best_val_acc": best_val_acc,
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "img_size": IMG_SIZE,
    }
    with open(args.output.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Done.", meta)


if __name__ == "__main__":
    main()
