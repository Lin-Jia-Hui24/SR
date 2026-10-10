from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def psnr(prediction: Tensor, target: Tensor, scale: int, shave: bool = True) -> float:
    if shave:
        prediction = prediction[..., scale:-scale, scale:-scale]
        target = target[..., scale:-scale, scale:-scale]
    mse = torch.mean((prediction.clamp(0, 1) - target) ** 2)
    return float((-10 * torch.log10(mse)).item()) if mse > 0 else float("inf")


def save_checkpoint(path: Path, **state: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
