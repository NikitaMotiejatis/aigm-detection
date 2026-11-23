"""The small CNN of Deezer's deepfake-detector (SimpleSpectrogramCNN with conf/base.json).

Written in PyTorch from that architecture, with a two-class head instead of one sigmoid.
"""

from __future__ import annotations

import torch
from torch import nn

HIDDEN = (16, 32, 64, 128, 256, 512)


class SmallCNN(nn.Module):
    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        if pretrained:
            raise ValueError("the small CNN has no pretrained weights")
        layers: list[nn.Module] = []
        channels = 1
        for hidden in HIDDEN:  # upstream order: convolution, ReLU, pooling, batch norm
            layers += [nn.Conv2d(channels, hidden, 3, padding=1), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2), nn.BatchNorm2d(hidden)]  # fmt: skip
            channels = hidden
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(channels, 128), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(128, 2),
        )  # fmt: skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).mean(dim=(2, 3)))
