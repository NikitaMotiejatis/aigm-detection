"""ResNet18 as in the benchmark: ImageNet weights, new one-channel conv1, two-class head."""

from __future__ import annotations

from torch import nn
from torchvision import models


def resnet18(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model
