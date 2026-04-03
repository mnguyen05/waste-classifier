"""Shared ResNet-18 binary head for train and inference."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_resnet18_binary(num_classes: int = 2, weights=None) -> nn.Module:
    m = models.resnet18(weights=weights)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, num_classes)
    return m


def load_pretrained_resnet18():
    """ImageNet weights enum; caller handles download failures."""
    return models.ResNet18_Weights.IMAGENET1K_V1
