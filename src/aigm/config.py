"""Configuration loading: one YAML file plus `key=value` overrides (PLAN.md decision 11)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> DictConfig:
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    return cfg
