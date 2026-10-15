"""Paired x2 augmentation for RGB float32 HWC images in [0, 1].

Consistency is a heuristic, not a registration probability. Local blending
keeps the HR target unchanged and never realigns either image.
"""

import numpy as np
from PIL import Image


def downsample_hr(hr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Antialiased float Lanczos reference; size=(height, width), no uint8 step."""
    height, width = size
    channels = []
    for channel in range(3):
        image = Image.fromarray(hr[..., channel].astype(np.float32))
        channels.append(np.asarray(image.resize((width, height), Image.Resampling.LANCZOS)))
    return np.clip(np.stack(channels, axis=-1), 0.0, 1.0)


def _smooth(array: np.ndarray) -> np.ndarray:
    """Apply the same five-tap low-pass filter vertically and horizontally."""
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
    height, width = array.shape
    padded = np.pad(array, ((2, 2), (0, 0)), mode="reflect")
    vertical = np.zeros_like(array)
    for k, weight in enumerate(kernel):
        vertical += weight * padded[k:k + height]
    padded = np.pad(vertical, ((0, 0), (2, 2)), mode="reflect")
    result = np.zeros_like(array)
    for k, weight in enumerate(kernel):
        result += weight * padded[:, k:k + width]
    return result


def _gradient_field(image: np.ndarray) -> np.ndarray:
    gray = np.asarray(image, dtype=np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gy, gx = np.gradient(_smooth(gray))
    return np.stack((gy, gx), axis=-1)


def _consistency_details(lr: np.ndarray, reference: np.ndarray) -> dict:
    """Compare normalized gradients at offsets ±2; blur/noise also affect this score."""
    if min(lr.shape[:2]) < 12:
        return {"score": None, "reason": "region_too_small"}
    fields = (_gradient_field(lr), _gradient_field(reference))
    margin = 3
    base = fields[0][margin:-margin, margin:-margin]
    base = base - base.mean(axis=(0, 1), keepdims=True)
    base_norm = float(np.linalg.norm(base))
    base_energy = float(np.sqrt(np.mean(base**2)))
    reference_core = fields[1][margin:-margin, margin:-margin]
    reference_energy = float(np.sqrt(np.mean((reference_core - reference_core.mean(axis=(0, 1))) ** 2)))
    if min(base_energy, reference_energy) < 5e-4:
        return {"score": None, "reason": "insufficient_structure"}
    height, width = fields[1].shape[:2]
    correlations = {}
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            shifted = fields[1][margin + dy:height - margin + dy, margin + dx:width - margin + dx]
            shifted = shifted - shifted.mean(axis=(0, 1), keepdims=True)
            norm = float(np.linalg.norm(shifted))
            correlation = float(np.sum(base * shifted) / max(base_norm * norm, 1e-12))
            correlations[(dy, dx)] = float(np.clip(correlation, -1.0, 1.0))
    zero = correlations[(0, 0)]
    # Prefer zero shift in ties, then the nearest shift, for stable diagnostics.
    best_offset = max(
        correlations,
        key=lambda offset: (correlations[offset], -(offset[0] ** 2 + offset[1] ** 2)),
    )
    best = correlations[best_offset]
    shift_advantage = max(0.0, best - zero)
    score = float(np.clip(max(0.0, zero) * np.exp(-4.0 * shift_advantage), 0.0, 1.0))
    return {
        "score": score,
        "reason": None,
        "zero_shift_correlation": zero,
        "best_shift_correlation": best,
        "best_offset_dy_dx": [int(value) for value in best_offset],
        "gradient_energy": [base_energy, reference_energy],
    }


def consistency_score(lr: np.ndarray, reference: np.ndarray) -> float | None:
    """Return a heuristic in [0, 1], or None for small/flat regions; inputs have equal size."""
    return _consistency_details(lr, reference)["score"]


def _geometry(array: np.ndarray, k: int, flip: bool) -> np.ndarray:
    result = np.rot90(array, k=k, axes=(0, 1))
    if flip:
        result = np.flip(result, axis=1)
    return np.ascontiguousarray(result, dtype=np.float32).copy()


def augment_pair(lr: np.ndarray, hr: np.ndarray, rng: np.random.Generator,
                 mode: str = "geometry", mix_strength: float = 0.3,
                 apply_probability: float = 0.5) -> tuple[np.ndarray, np.ndarray, dict]:
    """Shared D4, followed optionally by local blend or consistency-gated blend.

    HR changes only through geometry. ``applied`` reports mixing, not geometry.
    Probability 0.5 keeps half of samples unmixed in expectation, not per batch.
    """
    if mode not in {"geometry", "blend", "consistent_blend"}:
        raise ValueError(f"Unknown augmentation mode: {mode}")
    k = int(rng.integers(0, 4))
    flip = bool(rng.integers(0, 2))
    output_lr, output_hr = _geometry(lr, k, flip), _geometry(hr, k, flip)
    meta = {
        "mode": mode,
        "geometry": {"rot90_k": k, "horizontal_flip": flip},
        "applied": False, "apply_probability": float(apply_probability),
        "alpha": 0.0, "mask_bbox": None, "mix_fraction": 0.0,
        "effective_strength": 0.0, "consistency_score": None,
        "consistency_kind": "lowpass_gradient_heuristic_not_probability",
        "consistency_details": None, "skip_reason": None,
    }
    height, width = output_lr.shape[:2]
    if mode == "geometry":
        meta["skip_reason"] = "geometry_only"
    elif mix_strength == 0:
        meta["skip_reason"] = "zero_strength"
    elif rng.random() >= apply_probability:
        meta["skip_reason"] = "probability_draw"
    elif min(height, width) < 8:
        meta["skip_reason"] = "image_too_small"
    if meta["skip_reason"]:
        return output_lr, output_hr, meta
    region_height = min(height, max(8, int(height * rng.uniform(0.35, 0.75))))
    region_width = min(width, max(8, int(width * rng.uniform(0.35, 0.75))))
    y = int(rng.integers(0, height - region_height + 1))
    x = int(rng.integers(0, width - region_width + 1))
    meta["mask_bbox"] = [x, y, x + region_width, y + region_height]
    reference = downsample_hr(output_hr, (height, width))
    score = 1.0
    if mode == "consistent_blend":
        region = np.s_[y:y + region_height, x:x + region_width]
        details = _consistency_details(output_lr[region], reference[region])
        meta["consistency_details"] = details
        meta["consistency_score"] = details["score"]
        if details["score"] is None:
            meta["skip_reason"] = details["reason"]
            return output_lr, output_hr, meta
        score = details["score"]
        if score <= 1e-6:
            meta["skip_reason"] = "zero_consistency"
            return output_lr, output_hr, meta
    # Sine-squared taper avoids a rectangular discontinuity at the boundary.
    window_y = np.sin(np.pi * (np.arange(region_height) + 0.5) / region_height) ** 2
    window_x = np.sin(np.pi * (np.arange(region_width) + 0.5) / region_width) ** 2
    mask = np.zeros((height, width), dtype=np.float32)
    mask[y : y + region_height, x : x + region_width] = np.outer(window_y, window_x)
    weights = (float(mix_strength) * score * mask)[..., None]
    output_lr = np.clip((1 - weights) * output_lr + weights * reference, 0, 1).astype(np.float32)
    meta.update(
        applied=True,
        alpha=float(mix_strength),
        mix_fraction=float(region_height * region_width / (height * width)),
        effective_strength=float(weights.mean()),
    )
    return output_lr, output_hr, meta


def paired_mixup(lr: np.ndarray, hr: np.ndarray, other_lr: np.ndarray, other_hr: np.ndarray,
                 rng: np.random.Generator, beta: float = 0.2,
                 lambda_value: float | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Mix equally sized pairs with one shared lambda on both LR and HR."""
    lam = float(rng.beta(beta, beta) if lambda_value is None else lambda_value)
    mixed_lr = (lam * lr + (1.0 - lam) * other_lr).astype(np.float32)
    mixed_hr = (lam * hr + (1.0 - lam) * other_hr).astype(np.float32)
    return mixed_lr, mixed_hr, {"mode": "paired_mixup", "lambda": lam, "beta": float(beta)}
