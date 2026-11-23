"""Detectors: waveform in, two logits out, with the log-mel frontend inside (decision 3)."""

from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from aigm.models.frontend import LogMel
from aigm.models.mobilenet import mobilenet_v2
from aigm.models.resnet import resnet18
from aigm.models.smallcnn import SmallCNN

BACKBONES = {"resnet18": resnet18, "mobilenet_v2": mobilenet_v2, "smallcnn": SmallCNN}


class Detector(nn.Module):
    def __init__(self, frontend: nn.Module, backbone: nn.Module) -> None:
        super().__init__()
        self.frontend, self.backbone = frontend, backbone

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.frontend(wave))


def detection_score(logits: torch.Tensor) -> torch.Tensor:
    """Log-odds of "generated". Unlike a softmax it does not saturate into ties near 0 and 1."""
    return logits[:, 1] - logits[:, 0]


def build_model(cfg: DictConfig, pretrained: bool | None = None) -> Detector:
    """`pretrained=False` skips the ImageNet download when a checkpoint is loaded afterwards."""
    name = cfg.model.name
    if name not in BACKBONES:
        raise ValueError(f"unknown model '{name}', expected one of {sorted(BACKBONES)}")
    use_weights = cfg.model.get("pretrained", False) if pretrained is None else pretrained
    backbone = BACKBONES[name](pretrained=bool(use_weights))
    return Detector(LogMel(**OmegaConf.to_container(cfg.frontend, resolve=True)), backbone)
