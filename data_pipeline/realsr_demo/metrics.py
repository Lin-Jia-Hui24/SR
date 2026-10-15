"""One explicit RGB metric protocol, shared by Bicubic and the trained model."""
import numpy as np
import torch
from torch.nn import functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

PROTOCOL = {
    "color": "RGB", "data_range": 1.0, "extra_border_crop": 0,
    "aggregation": "per-image RGB PSNR/SSIM, then arithmetic mean",
    "psnr": "10*log10(1 / mean((prediction-target)**2)), MSE over all H*W*3 values",
    "prediction": "float32, clipped to [0,1], no uint8 quantization",
    "bicubic": "torch.interpolate(scale_factor=2, mode=bicubic, align_corners=False, antialias=False)",
    "ssim": "mean of R/G/B SSIM; gaussian 11x11, sigma=1.5, K1=0.01, K2=0.03, population covariance; valid-window mean excludes 5 pixels per side intrinsically",
    "test_augmentation": "none; no self ensemble, alignment, rescaling, color correction, or test-time fitting",
    "metric_precision": "float64 arithmetic after float32 inference; source RGB8 divided by 255",
}

def bicubic_x2(lr):
    tensor = torch.from_numpy(np.ascontiguousarray(lr.transpose(2, 0, 1))).unsqueeze(0)
    with torch.no_grad():
        out = F.interpolate(tensor, scale_factor=2, mode="bicubic", align_corners=False, antialias=False)
    return out[0].permute(1, 2, 0).numpy().clip(0, 1)

def rgb_metrics(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[2] != 3:
        raise ValueError("Metric inputs must have the same HWC RGB shape")
    prediction = np.asarray(prediction, dtype=np.float64).clip(0, 1)
    target = np.asarray(target, dtype=np.float64)
    return {
        "psnr_rgb": float(peak_signal_noise_ratio(target, prediction, data_range=1.0)),
        "ssim_rgb": float(structural_similarity(target, prediction, data_range=1.0,
            channel_axis=-1, gaussian_weights=True, win_size=11, sigma=1.5,
            K1=0.01, K2=0.03, use_sample_covariance=False)),
    }

