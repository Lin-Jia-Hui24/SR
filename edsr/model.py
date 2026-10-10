from __future__ import annotations

import math

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """EDSR residual block without batch normalization."""

    def __init__(self, channels: int, res_scale: float = 0.1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class Upsampler(nn.Sequential):
    def __init__(self, scale: int, channels: int) -> None:
        layers: list[nn.Module] = []
        if scale in (2, 4, 8):
            for _ in range(int(math.log2(scale))):
                layers.extend([
                    nn.Conv2d(channels, 4 * channels, 3, padding=1),
                    nn.PixelShuffle(2),
                ])
        elif scale == 3:
            layers.extend([
                nn.Conv2d(channels, 9 * channels, 3, padding=1),
                nn.PixelShuffle(3),
            ])
        else:
            raise ValueError("scale must be 2, 3, 4, or 8")
        super().__init__(*layers)


class EDSR(nn.Module):
    """Enhanced Deep Super-Resolution network (EDSR)."""

    def __init__(
        self,
        scale: int = 4,
        num_features: int = 256,
        num_blocks: int = 32,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.head = nn.Conv2d(3, num_features, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(num_features, res_scale) for _ in range(num_blocks)])
        self.body_tail = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.tail = nn.Sequential(
            Upsampler(scale, num_features),
            nn.Conv2d(num_features, 3, 3, padding=1),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        residual = self.body_tail(self.body(x))
        return self.tail(x + residual)
