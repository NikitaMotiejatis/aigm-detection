"""Mono mix-down and resampling to the working rate."""

from __future__ import annotations

import numpy as np
import soxr


def to_mono(x: np.ndarray) -> np.ndarray:
    """[frames, channels] -> [frames], the plain mean of the channels, as float64."""
    return x.astype(np.float64).mean(axis=1) if x.ndim == 2 else x.astype(np.float64)


def resample(x: np.ndarray, rate: int, target_rate: int, quality: str = "VHQ") -> np.ndarray:
    """Resample with soxr. Input that is already at the target rate is returned untouched."""
    if rate == target_rate:
        return x
    return soxr.resample(x, rate, target_rate, quality=quality)
