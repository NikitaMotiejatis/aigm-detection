"""Log-mel frontend, computed on the model's device (WP2 step 1).

Follows the benchmark's librosa pipeline (Slaney mel, power to dB relative to the clip's maximum,
80 dB range, bilinear resize to 224 x 224), except that the band stops at 7 kHz (decision 2) and
each clip is standardised (WP2 step 1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

AMIN = 1e-10


class LogMel(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: float = 20.0,
        f_max: float = 7000.0,
        top_db: float = 80.0,
        size: int = 224,
    ) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
            f_min=f_min, f_max=f_max, power=2.0, norm="slaney", mel_scale="slaney",
            pad_mode="constant",
        )  # fmt: skip
        self.top_db, self.size = top_db, size

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """[batch, samples] -> [batch, 1, size, size]"""
        with torch.autocast(device_type=wave.device.type, enabled=False):
            power = self.mel(wave.float()).clamp_min(AMIN)
            db = 10 * torch.log10(power / power.amax(dim=(1, 2), keepdim=True))
            x = db.clamp_min(-self.top_db).unsqueeze(1)
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
            mean = x.mean(dim=(2, 3), keepdim=True)
            std = x.std(dim=(2, 3), keepdim=True)
            return (x - mean) / std.clamp_min(1e-6)
