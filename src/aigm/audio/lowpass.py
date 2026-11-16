"""The common band limit (PLAN.md decision 2).

Real clips reach 16 kHz through our resampler, generated clips arrive at 16 kHz from somebody
else's. The top of the band (roughly 7.3-8 kHz) therefore carries the fingerprint of the
resampling filter, which differs by class. Every clip of every source goes through the same
linear-phase FIR low-pass that removes that region.
"""

from __future__ import annotations

from functools import cache

import numpy as np
from scipy import signal


@cache
def design(rate: int, passband_hz: float, stopband_hz: float, attenuation_db: float) -> np.ndarray:
    """Kaiser-window FIR, odd length; the -6 dB point is midway between the two edges."""
    nyquist = rate / 2
    numtaps, beta = signal.kaiserord(attenuation_db, (stopband_hz - passband_hz) / nyquist)
    numtaps |= 1
    cutoff = (passband_hz + stopband_hz) / 2
    return signal.firwin(numtaps, cutoff, window=("kaiser", beta), fs=rate)


def apply(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Zero-phase filtering of one clip; the clip is zero-extended at both ends."""
    return np.convolve(x, taps, mode="same")
