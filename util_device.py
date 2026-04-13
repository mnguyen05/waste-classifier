from __future__ import annotations
import torch

def get_torch_device() -> torch.device:
    return torch.device("cpu")
