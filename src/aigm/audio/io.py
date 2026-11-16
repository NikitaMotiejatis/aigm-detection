"""Decoding of audio files to float32 arrays at their native sample rate.

WAV and FLAC are read with libsndfile. Everything else (MP3) is decoded by the ffmpeg binary
shipped with `imageio-ffmpeg`, so that both machines decode with the same binary.
"""

from __future__ import annotations

import struct
import subprocess
from functools import cache
from pathlib import Path

import numpy as np
import soundfile as sf

SNDFILE_SUFFIXES = {".wav", ".flac"}


class DecodeError(RuntimeError):
    pass


@cache
def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@cache
def ffmpeg_version() -> str:
    out = subprocess.run([ffmpeg_exe(), "-version"], capture_output=True, text=True).stdout
    return out.splitlines()[0] if out else "unknown"


def _parse_wav(buf: bytes) -> tuple[np.ndarray, int]:
    """Parse a float32 WAV stream written by ffmpeg to a pipe (its length fields are not valid)."""
    if buf[:4] != b"RIFF" or buf[8:12] != b"WAVE":
        raise DecodeError("ffmpeg did not return a WAV stream")
    pos, channels, rate = 12, 0, 0
    while pos + 8 <= len(buf):
        chunk, size = buf[pos : pos + 4], struct.unpack_from("<I", buf, pos + 4)[0]
        if chunk == b"fmt ":
            channels, rate = struct.unpack_from("<HI", buf, pos + 10)
        elif chunk == b"data":
            if not channels:
                break
            payload = memoryview(buf)[pos + 8 :]
            frames = len(payload) // (4 * channels)
            data = np.frombuffer(payload[: frames * 4 * channels], dtype="<f4")
            return data.reshape(frames, channels), rate
        pos += 8 + size + (size & 1)
    raise DecodeError("no audio data in the ffmpeg output")


def decode(path: str | Path) -> tuple[np.ndarray, int]:
    """Return (samples of shape [frames, channels] as float32, sample rate)."""
    path = Path(path)
    if path.suffix.lower() in SNDFILE_SUFFIXES:
        try:
            data, rate = sf.read(path, dtype="float32", always_2d=True)
        except sf.LibsndfileError as exc:
            raise DecodeError(str(exc)) from exc
        return data, rate
    cmd = [
        ffmpeg_exe(), "-v", "error", "-nostdin", "-i", str(path),
        "-map", "0:a:0", "-map_metadata", "-1", "-bitexact",
        "-c:a", "pcm_f32le", "-f", "wav", "-",
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True)
    if not proc.stdout:
        message = proc.stderr.decode(errors="replace").strip().splitlines()
        raise DecodeError(message[-1] if message else f"ffmpeg exit code {proc.returncode}")
    return _parse_wav(proc.stdout)
