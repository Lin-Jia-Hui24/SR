"""Checkpoint loading utilities for local and published EDSR weights."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn


def _find_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    """Extract tensors from common checkpoint layouts without guessing their config."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a dictionary containing model tensors.")
    for key in ("model", "state_dict", "model_state_dict", "params", "params_ema"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping) and any(isinstance(value, torch.Tensor) for value in candidate.values()):
            return candidate  # type: ignore[return-value]
    if any(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint  # type: ignore[return-value]
    raise KeyError("Could not find a state dict. Expected model, state_dict, or model_state_dict.")


def _normalise_key(key: str, num_blocks: int) -> str:
    """Translate the common EDSR-PyTorch module layout to this project's layout."""
    while key.startswith("module.") or key.startswith("model."):
        key = key.split(".", 1)[1]
    # sanghyun-son/EDSR-PyTorch: head is a one-element Sequential.
    if key.startswith("head.0."):
        key = "head." + key[len("head.0."):]
    # Its final body convolution lives at body.<number>, while ours names it body_tail.
    if key.startswith(f"body.{num_blocks}."):
        key = "body_tail." + key[len(f"body.{num_blocks}."):]
    return key


def load_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device | str,
    *,
    strict: bool = True,
    num_blocks: int | None = None,
) -> dict[str, object]:
    """Load weights and return a report; strict mode rejects architecture mismatches."""
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source = _find_state_dict(raw)
    blocks = num_blocks if num_blocks is not None else len(getattr(model, "body", []))
    translated = {_normalise_key(key, blocks): value for key, value in source.items()}
    target = model.state_dict()
    compatible = {key: value for key, value in translated.items() if key in target and value.shape == target[key].shape}
    incompatible_shapes = sorted(key for key, value in translated.items() if key in target and value.shape != target[key].shape)
    missing = sorted(set(target) - set(compatible))
    unexpected = sorted(set(translated) - set(compatible))
    if strict and (missing or unexpected):
        details = f"missing={len(missing)}, unexpected={len(unexpected)}, shape_mismatch={len(incompatible_shapes)}"
        raise RuntimeError(
            "Weights are incompatible with the configured EDSR architecture (" + details + "). "
            "Match scale, num_features, and num_blocks to the published checkpoint, or use --non-strict for transfer learning."
        )
    model.load_state_dict(compatible, strict=False)
    return {
        "checkpoint_path": str(checkpoint_path),
        "loaded": len(compatible),
        "model_tensors": len(target),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": incompatible_shapes,
    }
