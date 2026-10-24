from __future__ import annotations

import random
import os
import tempfile
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
    # A preempted worker must never replace a usable resume checkpoint with a
    # partially written archive.  Write beside the destination and atomically
    # publish it only after torch.save has completed.
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def self_ensemble_predict(model: torch.nn.Module, lr_tensor: Tensor) -> Tensor:
    """Self-ensemble x8: 4 rotations × 2 flips, average the predictions."""
    preds = []
    for flip in [False, True]:
        for rot in range(4):
            x = lr_tensor
            if flip:
                x = torch.flip(x, [-1])
            if rot > 0:
                x = torch.rot90(x, rot, [-2, -1])
            sr = model(x)
            if rot > 0:
                sr = torch.rot90(sr, -rot, [-2, -1])
            if flip:
                sr = torch.flip(sr, [-1])
            preds.append(sr)
    return torch.stack(preds).mean(dim=0)
