"""Conversion of the raw datasets into the processed cache (WP1 step 3).

decode -> mono -> resample to 16 kHz (skipped when already 16 kHz) -> cut into 10 s clips
-> common low-pass -> optional RMS normalisation -> clip to [-1, 1] -> 16-bit.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from aigm.audio import lowpass
from aigm.audio.io import DecodeError, decode, ffmpeg_version
from aigm.audio.resample import resample, to_mono
from aigm.data.cache import INDEX_COLUMNS
from aigm.data.sources import SOURCES


def clip_samples(audio: DictConfig) -> int:
    return int(audio.sample_rate * audio.clip_seconds)


def load_mono(path: Path, audio: DictConfig) -> np.ndarray:
    """Decode a file and return it as mono float64 at the working rate."""
    data, rate = decode(path)
    return resample(to_mono(data), rate, audio.sample_rate, audio.resample_quality)


def window_starts(length: int, clip: int, count: int) -> list[int]:
    """Start samples of up to `count` evenly spaced clips; one clip if the audio is shorter.

    A track that misses a whole number of clips by under 5 % of a clip still gets that number
    (a 29.98 s track gives three clips that overlap by a few milliseconds).
    """
    if length <= clip:
        return [0]
    n = min(count, max(1, (length + clip // 20) // clip))
    return [int(round(s)) for s in np.linspace(0, length - clip, n)]


def finish_clip(x: np.ndarray, audio: DictConfig) -> np.ndarray:
    """Pad or crop to the clip length, band-limit, normalise if configured, quantise to int16.

    A clip of pure digital silence has no level to normalise and stays silent.
    """
    clip = clip_samples(audio)
    x = x[:clip]
    if len(x) < clip:
        x = np.pad(x, (0, clip - len(x)))
    lp = audio.lowpass
    taps = lowpass.design(audio.sample_rate, lp.passband_hz, lp.stopband_hz, lp.attenuation_db)
    x = lowpass.apply(x, taps)
    if audio.normalise_rms:
        rms = float(np.sqrt(np.mean(x**2)))
        if rms > 0:
            x = x * (10 ** (audio.rms_target_dbfs / 20) / rms)
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)


def convert_file(
    path: Path, audio: DictConfig, count: int = 1, min_seconds: float = 0.0
) -> tuple[np.ndarray, list[int], list[int]]:
    """Convert one raw file to (clips [n, clip_samples] int16, start samples, valid samples)."""
    x = load_mono(path, audio)
    if len(x) == 0:
        raise DecodeError("no audio samples")
    if len(x) < min_seconds * audio.sample_rate:
        raise DecodeError(f"too short: {len(x) / audio.sample_rate:.2f} s")
    clip = clip_samples(audio)
    starts = window_starts(len(x), clip, count)
    clips = np.stack([finish_clip(x[s : s + clip], audio) for s in starts])
    return clips, starts, [min(clip, len(x) - s) for s in starts]


_worker_state: dict = {}


def _init_worker(raw: str, audio: dict, count: int, min_seconds: float) -> None:
    _worker_state.update(raw=Path(raw), audio=OmegaConf.create(audio), count=count,
                         min_seconds=min_seconds)  # fmt: skip


def _convert_job(rel_path: str):
    s = _worker_state
    try:
        return convert_file(s["raw"] / rel_path, s["audio"], s["count"], s["min_seconds"]), None
    except DecodeError as exc:
        return None, str(exc)


def _git_commit() -> str:
    """HEAD, with `-dirty` appended when the working tree differs from it."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    except OSError:
        return "unknown"
    commit = head.stdout.strip() or "unknown"
    return commit + "-dirty" if status.stdout.strip() else commit


def prepare(source: str, cfg: DictConfig, force: bool = False, workers: int | None = None) -> dict:
    """Build the cache files of one source. Refuses to replace an existing cache without `force`."""
    raw, processed = Path(cfg.raw_dir), Path(cfg.processed_dir)
    final = {kind: processed / f"{source}.{kind}"
             for kind in ("i16", "index.csv", "meta.json", "skipped.csv")}  # fmt: skip
    existing = [str(p) for p in final.values() if p.exists()]
    if existing and not force:
        raise FileExistsError(
            f"the cache for '{source}' exists ({existing[0]}). It is built once and must not be "
            "rebuilt after the splits are frozen; pass --force only if a rebuild is intended."
        )
    table = SOURCES[source](raw)
    window = cfg.windows.get(source, {"count": 1, "min_seconds": 0})
    audio = OmegaConf.to_container(cfg.audio, resolve=True)
    clip = clip_samples(cfg.audio)
    processed.mkdir(parents=True, exist_ok=True)
    tmp_array = processed / f"{source}.i16.tmp"
    rows, skipped = [], []
    ctx = multiprocessing.get_context("spawn")
    init = (str(raw), audio, int(window["count"]), float(window["min_seconds"]))
    with (
        ctx.Pool(workers or os.cpu_count(), _init_worker, init) as pool,
        open(tmp_array, "wb") as out,
    ):
        results = pool.imap(_convert_job, table["path"], chunksize=4)
        jobs = zip(table.itertuples(), results, strict=True)
        for i, (item, (result, error)) in enumerate(jobs, 1):
            if error is not None:
                skipped.append({"path": item.path, "reason": error})
            else:
                clips, starts, valid = result
                out.write(clips.tobytes())
                for w, (start, n_valid) in enumerate(zip(starts, valid, strict=True)):
                    single = window["count"] == 1
                    rows.append({
                        "row": len(rows),
                        "clip_id": item.item_id if single else f"{item.item_id}/{w}",
                        "source": item.source, "label": item.label, "generator": item.generator,
                        "group_id": item.group_id, "track_id": item.item_id,
                        "variant": item.variant, "path": item.path,
                        "offset_s": round(start / cfg.audio.sample_rate, 4),
                        "valid_samples": n_valid,
                    })  # fmt: skip
            if i % 2000 == 0 or i == len(table):
                print(f"{source}: {i}/{len(table)} files, {len(rows)} clips, "
                      f"{len(skipped)} skipped", flush=True)  # fmt: skip
    meta = {
        "source": source, "clips": len(rows), "clip_samples": clip, "dtype": "int16",
        "files": len(table), "skipped": len(skipped), "audio": audio, "window": dict(window),
        "versions": {"soxr": version("soxr"), "scipy": version("scipy"),
                     "numpy": version("numpy"), "ffmpeg": ffmpeg_version()},
        "git_commit": _git_commit(), "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }  # fmt: skip
    pd.DataFrame(rows, columns=INDEX_COLUMNS).to_csv(final["index.csv"], index=False)
    pd.DataFrame(skipped, columns=["path", "reason"]).to_csv(final["skipped.csv"], index=False)
    final["meta.json"].write_text(json.dumps(meta, indent=2) + "\n")
    tmp_array.replace(final["i16"])
    return meta
