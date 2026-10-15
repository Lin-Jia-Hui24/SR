"""Small residual RGB x2 network with a bicubic skip and PixelShuffle head."""
from torch import nn
from torch.nn import functional as F


class TinySR(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.head = nn.Conv2d(3, width, 3, padding=1)
        self.body = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(width, width, 3, padding=1))
        self.up = nn.Sequential(nn.Conv2d(width, 12, 3, padding=1), nn.PixelShuffle(2))
        nn.init.zeros_(self.up[0].weight)
        nn.init.zeros_(self.up[0].bias)

    def forward(self, lr):
        features = self.head(lr)
        residual = self.up(features + self.body(features))
        return F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False,
                             antialias=False) + residual
