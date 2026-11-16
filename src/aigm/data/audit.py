"""Audit of the raw audio (WP1 step 2): what each source looks like before conversion.

Answers two questions before the cache is built: whether the common low-pass sits below the
band limit of every source, and whether loudness differs systematically between the classes.
"""

from __future__ import annotations

import multiprocessing
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy import signal

from aigm.audio.io import DecodeError, decode
from aigm.audio.resample import to_mono
from aigm.data.sources import SOURCES

GRID_HZ = np.arange(0.0, 24000.0, 25.0)
SILENCE_DBFS = -60.0
SPECTRUM_SECONDS = 30


def _db(x: float) -> float:
    return float(10 * np.log10(max(x, 1e-20)))


def measure(path: Path) -> tuple[dict, np.ndarray]:
    """Statistics of one raw file and its power spectrum on the common frequency grid."""
    data, rate = decode(path)
    x = to_mono(data)
    if len(x) == 0:
        raise DecodeError("no audio samples")
    power = float(np.mean(x**2))
    part = x[: rate * SPECTRUM_SECONDS]
    freqs, psd = signal.welch(part, fs=rate, nperseg=min(len(part), int(rate * 0.064)))
    above = np.nonzero(psd >= psd.max() * 1e-6)[0]
    spectrum = np.interp(GRID_HZ, freqs, psd, right=np.nan)
    stats = {
        "sample_rate": rate, "channels": data.shape[1], "duration_s": len(x) / rate,
        "peak": float(np.abs(data).max()), "rms_dbfs": _db(power),
        "silent": _db(power) < SILENCE_DBFS,
        "cutoff_hz": float(freqs[above[-1]]) if len(above) else 0.0,
    }  # fmt: skip
    return stats, spectrum


def _job(args: tuple[str, str]):
    raw, rel_path = args
    try:
        return measure(Path(raw) / rel_path)
    except DecodeError:
        return None


def _values(series: pd.Series) -> str:
    return "/".join(str(v) for v in sorted(series.unique()))


def audit(cfg: DictConfig, out_dir: Path, workers: int | None = None) -> pd.DataFrame:
    raw = Path(cfg.raw_dir)
    rng = np.random.default_rng(cfg.audit.seed)
    sample = []
    for reader in SOURCES.values():
        for _, group in reader(raw).groupby(["source", "generator"], sort=True):
            n = min(len(group), cfg.audit.clips_per_group)
            sample.append(group.iloc[np.sort(rng.choice(len(group), n, replace=False))])
    sample = pd.concat(sample, ignore_index=True)
    jobs = [(str(raw), p) for p in sample["path"]]
    with multiprocessing.get_context("spawn").Pool(workers or os.cpu_count()) as pool:
        results = pool.map(_job, jobs, chunksize=8)
    ok = [r is not None for r in results]
    stats = pd.DataFrame([r[0] for r in results if r is not None])
    spectra = np.stack([r[1] for r in results if r is not None])
    stats = pd.concat([sample[ok].reset_index(drop=True), stats], axis=1)

    rows, curves = [], {}
    for (source, generator), g in stats.groupby(["source", "generator"], sort=True):
        failed = int(((sample["source"] == source) & (sample["generator"] == generator)).sum())
        rows.append({
            "source": source, "generator": generator, "label": int(g["label"].iloc[0]),
            "clips": len(g), "undecodable": failed - len(g),
            "sample_rate": _values(g["sample_rate"]), "channels": _values(g["channels"]),
            "duration_mean_s": g["duration_s"].mean(), "duration_min_s": g["duration_s"].min(),
            "duration_max_s": g["duration_s"].max(),
            "peak_mean": g["peak"].mean(), "peak_max": g["peak"].max(),
            "rms_dbfs_mean": g["rms_dbfs"].mean(), "rms_dbfs_std": g["rms_dbfs"].std(),
            "rms_dbfs_median": g["rms_dbfs"].median(), "silent_fraction": g["silent"].mean(),
            "cutoff_hz_p05": g["cutoff_hz"].quantile(0.05),
            "cutoff_hz_median": g["cutoff_hz"].median(),
            "cutoff_hz_p95": g["cutoff_hz"].quantile(0.95),
        })  # fmt: skip
        with warnings.catch_warnings():  # bins above a source's Nyquist frequency are all NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            curves[f"{source} / {generator}"] = np.nanmean(spectra[g.index.to_numpy()], axis=0)
    summary = pd.DataFrame(rows).round(3)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "audit.csv", index=False)
    _plot(curves, out_dir / "audit_spectra.pdf", cfg)
    return summary


def _plot(curves: dict[str, np.ndarray], path: Path, cfg: DictConfig) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted({name.split(" / ")[0] for name in curves})
    fig, axes = plt.subplots(len(names), 1, figsize=(8, 3 * len(names)), sharex=True)
    for ax, source in zip(np.atleast_1d(axes), names, strict=True):
        for name, curve in curves.items():
            if name.startswith(source + " / "):
                with np.errstate(divide="ignore"):
                    ax.plot(GRID_HZ / 1000, 10 * np.log10(curve), lw=1, label=name.split(" / ")[1])
        for edge in (cfg.audio.lowpass.passband_hz, cfg.audio.lowpass.stopband_hz):
            ax.axvline(edge / 1000, color="grey", lw=0.6, ls="--")
        ax.set_title(source, loc="left", fontsize=10)
        ax.set_ylabel("mean PSD, dB")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=3)
    np.atleast_1d(axes)[-1].set_xlabel("frequency, kHz (dashed: edges of the common low-pass)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
