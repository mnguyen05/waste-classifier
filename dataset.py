"""Load images from the Kaggle-style folder layout with binary labels."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from label_mapping import CATEGORY_TO_LABEL, category_to_index

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _find_images_root(data_root: Path) -> Path:
    """Resolve .../images or .../images/images to the parent of category folders."""
    data_root = data_root.resolve()
    nested = data_root / "images"
    if nested.is_dir():
        if (nested / "images").is_dir():
            return nested / "images"
        return nested
    return data_root


class WasteBinaryDataset(Dataset):
    def __init__(self, paths: list[Path], labels: list[int], transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def collect_samples(images_root: Path) -> tuple[list[Path], list[int]]:
    root = _find_images_root(images_root)
    paths: list[Path] = []
    labels: list[int] = []

    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        if category not in CATEGORY_TO_LABEL:
            raise KeyError(
                f"Unknown category folder {category!r}. Add it to label_mapping.CATEGORY_TO_LABEL."
            )
        label = category_to_index(category)
        for sub in ("default", "real_world"):
            subdir = category_dir / sub
            if not subdir.is_dir():
                continue
            for p in sorted(subdir.iterdir()):
                if p.suffix.lower() in IMAGE_SUFFIXES:
                    paths.append(p)
                    labels.append(label)

    if not paths:
        raise FileNotFoundError(
            f"No images under {root}. Expected category folders with default/ and real_world/."
        )
    return paths, labels
