"""MobileNetV2 as in the benchmark: ImageNet weights, one-channel input, two-class head."""

from __future__ import annotations

from torch import nn
from torchvision import models


def mobilenet_v2(pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    # upstream replaces the whole first block, so its BatchNorm and ReLU are gone too
    model.features[0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model
