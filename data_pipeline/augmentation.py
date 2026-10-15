"""训练集的成对 ×2 增强：输入、输出均为 RGB [3,H,W] Tensor。"""

import torch


def augment_pair(lr, hr, patch_size=0):
    """同步随机裁剪、翻转和旋转，返回 (lr, hr)。

    patch_size=0 保留整图；正数表示 LR 方形裁剪的边长。
    只在训练集调用。用 torch.manual_seed 在训练开始时设置随机种子。
    """
    if patch_size:
        height, width = lr.shape[-2:]
        top = int(torch.randint(height - patch_size + 1, ()))
        left = int(torch.randint(width - patch_size + 1, ()))
        lr = lr[:, top:top + patch_size, left:left + patch_size]
        hr = hr[:, 2 * top:2 * (top + patch_size), 2 * left:2 * (left + patch_size)]

    if torch.rand(()) < 0.5:
        lr, hr = lr.flip(-1), hr.flip(-1)
    if torch.rand(()) < 0.5:
        lr, hr = lr.flip(-2), hr.flip(-2)
    k = int(torch.randint(4, ()))
    lr = torch.rot90(lr, k, dims=(-2, -1))
    hr = torch.rot90(hr, k, dims=(-2, -1))
    return lr.contiguous(), hr.contiguous()
