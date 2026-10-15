"""训练集的成对 ×2 增强：RGB float Tensor，范围 [0,1]。

Mixup 参考 Yoo et al., CVPR 2020 的成对混合，使用 Beta(0.2, 0.2)。
hr_blend / consistent_blend 是受 CutBlur 启发的 LR 域实验变体，非原版复现。
论文与官方实现：https://github.com/clovaai/cutblur
"""

import torch
from torch.nn import functional as F


def _crop_pair(lr, hr, patch_size):
    if patch_size:
        height, width = lr.shape[-2:]
        top = int(torch.randint(height - patch_size + 1, ()))
        left = int(torch.randint(width - patch_size + 1, ()))
        lr = lr[:, top:top + patch_size, left:left + patch_size]
        hr = hr[:, 2 * top:2 * (top + patch_size), 2 * left:2 * (left + patch_size)]
    return lr, hr


def _geometry(image, horizontal, vertical, k):
    if horizontal:
        image = image.flip(-1)
    if vertical:
        image = image.flip(-2)
    return torch.rot90(image, k, dims=(-2, -1)).contiguous()


def _gradient_field(image):
    gray = (image * image.new_tensor([0.299, 0.587, 0.114])[:, None, None]).sum(0)
    kernel = image.new_tensor([1, 4, 6, 4, 1]) / 16
    kernel = torch.outer(kernel, kernel)[None, None]
    smooth = F.conv2d(F.pad(gray[None, None], (2, 2, 2, 2), mode="reflect"), kernel)[0, 0]
    return torch.stack(torch.gradient(smooth))


def _consistency_score(lr, reference):
    """低通梯度相似性启发式，非配准概率；平坦/过小区域返回 0。"""
    if min(lr.shape[-2:]) < 12:
        return 0.0
    first, second = _gradient_field(lr), _gradient_field(reference)
    base, core = first[:, 3:-3, 3:-3], second[:, 3:-3, 3:-3]
    base = base - base.mean(dim=(-2, -1), keepdim=True)
    core = core - core.mean(dim=(-2, -1), keepdim=True)
    if min(base.square().mean().sqrt().item(), core.square().mean().sqrt().item()) < 5e-4:
        return 0.0
    base_norm = torch.linalg.vector_norm(base)
    height, width = lr.shape[-2:]
    correlations = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            shifted = second[:, 3 + dy:height - 3 + dy, 3 + dx:width - 3 + dx]
            shifted = shifted - shifted.mean(dim=(-2, -1), keepdim=True)
            denominator = (base_norm * torch.linalg.vector_norm(shifted)).clamp_min(1e-12)
            correlations.append(((base * shifted).sum() / denominator).clamp(-1, 1))
    correlations = torch.stack(correlations)
    zero = correlations[12]
    penalty = torch.exp(-4 * (correlations.max() - zero).clamp_min(0))
    return float(zero.clamp_min(0) * penalty)


def _hr_blend(lr, hr, strength, gated):
    height, width = lr.shape[-2:]
    if min(height, width) < 8 or strength == 0:
        return lr
    rh = min(height, max(8, int(height * (0.35 + 0.4 * torch.rand(())))))
    rw = min(width, max(8, int(width * (0.35 + 0.4 * torch.rand(())))))
    y = int(torch.randint(height - rh + 1, ()))
    x = int(torch.randint(width - rw + 1, ()))
    reference = F.interpolate(hr[None], size=(height, width), mode="bicubic",
                              align_corners=False, antialias=True)[0].clamp(0, 1)
    source = lr[:, y:y + rh, x:x + rw]
    target = reference[:, y:y + rh, x:x + rw]
    score = _consistency_score(source, target) if gated else 1.0
    if score == 0:
        return lr
    wy = torch.sin(torch.pi * (torch.arange(rh, device=lr.device, dtype=lr.dtype) + 0.5) / rh).square()
    wx = torch.sin(torch.pi * (torch.arange(rw, device=lr.device, dtype=lr.dtype) + 0.5) / rw).square()
    weight = strength * score * torch.outer(wy, wx)
    result = lr.clone()
    result[:, y:y + rh, x:x + rw] = (1 - weight) * source + weight * target
    return result


def augment_pair(lr, hr, patch_size=0, *, mode="geometry", other=None, p=0.5, strength=0.3):
    """输入 CHW 的 LR/HR，返回两个同 dtype/device 的 Tensor。

    mode: none / geometry / mixup / hr_blend / consistent_blend。
    patch_size=0 保留整图；正数表示同步裁剪的 LR 边长。
    other=(lr2,hr2) 仅供 Mixup，必须来自另一个训练样本。
    p 控制额外混合概率，strength 控制 HR 引导局部混合强度。
    几何变换始终执行（none 除外）。建议在 CPU Dataset 中调用，
    用 torch.manual_seed 设置随机种子；验证/测试不调用本函数。
    """
    if mode not in {"none", "geometry", "mixup", "hr_blend", "consistent_blend"}:
        raise ValueError(f"Unknown augmentation mode: {mode}")
    if not 0 <= p <= 1 or not 0 <= strength <= 1:
        raise ValueError("p and strength must be in [0, 1]")
    if mode == "mixup":
        if other is None:
            raise ValueError("Mixup requires other=(lr2, hr2) from another training pair")
        if not patch_size and (lr.shape != other[0].shape or hr.shape != other[1].shape):
            raise ValueError("Mixup needs matching image sizes, or a positive patch_size")
    lr, hr = _crop_pair(lr, hr, patch_size)
    if mode == "none":
        return lr.contiguous(), hr.contiguous()
    horizontal, vertical = bool(torch.rand(()) < 0.5), bool(torch.rand(()) < 0.5)
    k = int(torch.randint(4, ()))
    lr, hr = _geometry(lr, horizontal, vertical, k), _geometry(hr, horizontal, vertical, k)
    if mode == "geometry" or p == 0 or torch.rand(()) >= p:
        return lr, hr
    if mode == "mixup":
        other_lr, other_hr = _crop_pair(*other, patch_size)
        other_lr = _geometry(other_lr, horizontal, vertical, k)
        other_hr = _geometry(other_hr, horizontal, vertical, k)
        lam = float(torch.distributions.Beta(0.2, 0.2).sample())
        lr, hr = lam * lr + (1 - lam) * other_lr, lam * hr + (1 - lam) * other_hr
    else:
        lr = _hr_blend(lr, hr, strength, gated=mode == "consistent_blend")
    return lr.contiguous(), hr.contiguous()
