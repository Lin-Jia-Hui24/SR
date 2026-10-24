from __future__ import annotations

import math

import torch
from torch import nn


class MeanShift(nn.Conv2d):
    """Frozen RGB mean shift used by the published full EDSR checkpoints."""

    def __init__(self, sign: int = -1) -> None:
        super().__init__(3, 3, kernel_size=1)
        # The official r32f256 EDSR checkpoints operate in RGB [0, 255].
        rgb_mean = 255.0 * torch.tensor([0.4488, 0.4371, 0.4040])
        with torch.no_grad():
            self.weight.copy_(torch.eye(3).view(3, 3, 1, 1))
            self.bias.copy_(sign * rgb_mean)
        self.weight.requires_grad_(False)
        self.bias.requires_grad_(False)


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
        scale: int = 2,
        num_features: int = 256,
        num_blocks: int = 32,
        res_scale: float = 0.1,
    ) -> None:
        super().__init__()
        # Callers use RGB [0, 1], while published EDSR weights use [0, 255]
        # around their fixed mean-shift layers. Convert only at this boundary.
        self.sub_mean = MeanShift(-1)
        self.head = nn.Conv2d(3, num_features, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(num_features, res_scale) for _ in range(num_blocks)])
        self.body_tail = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.tail = nn.Sequential(
            Upsampler(scale, num_features),
            nn.Conv2d(num_features, 3, 3, padding=1),
        )
        self.add_mean = MeanShift(1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and not isinstance(module, MeanShift):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(self.sub_mean(x * 255.0))
        residual = self.body_tail(self.body(x))
        return self.add_mean(self.tail(x + residual)) / 255.0


def build_model(cfg: dict) -> EDSR:
    """Instantiate an EDSR model from config dict (the ``model`` section of the YAML)."""
    return EDSR(
        scale=cfg.get("scale", 2),
        num_features=cfg.get("num_features", 256),
        num_blocks=cfg.get("num_blocks", 32),
        res_scale=cfg.get("res_scale", 0.1),
    )
