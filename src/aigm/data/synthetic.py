"""A tiny cache in the real on-disk format, for the smoke run (WP2 step 7). Never touches data/."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aigm.data.cache import INDEX_COLUMNS, write_manifest

SOURCE = "synthetic"
SAMPLES = 160_000
RATE = 16_000
PARTS = {"train": 40, "val": 12, "test": 12}


def _clip(rng: np.random.Generator, label: int) -> np.ndarray:
    t = np.arange(SAMPLES) / RATE
    f0 = rng.uniform(150, 900)
    harmonics = (1, 2, 3) if label == 0 else (1, 3, 5, 7)
    x = sum(np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6)) / h for h in harmonics)
    x = x + (0.02 if label == 0 else 0.15) * rng.standard_normal(SAMPLES)
    return np.round(0.25 * x / np.max(np.abs(x)) * 32767).astype(np.int16)


def ensure_cache(root: str | Path, seed: int = 0) -> Path:
    """Build the cache and split files once; the manifest is written last and marks completion."""
    root = Path(root)
    processed, splits = root / "processed", root / "splits"
    manifest = root / "CACHE.sha256"
    if manifest.exists():
        return manifest
    processed.mkdir(parents=True, exist_ok=True)
    splits.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, row = [], 0
    with open(processed / f"{SOURCE}.i16", "wb") as f:
        for part, count in PARTS.items():
            part_rows = []
            for i in range(count):
                label = i % 2
                generator = "gen_a" if label else "real"
                _clip(rng, label).tofile(f)
                clip_id = f"syn/{part}/{i:03d}"
                part_rows.append({"row": row, "clip_id": clip_id, "source": SOURCE, "label": label,
                                  "generator": generator, "group_id": f"{part}{i // 2}",
                                  "track_id": clip_id, "variant": "", "path": clip_id,
                                  "offset_s": 0.0, "valid_samples": SAMPLES})  # fmt: skip
                row += 1
            rows += part_rows
            pd.DataFrame(part_rows)[["clip_id", "source", "label", "generator", "group_id"]].to_csv(
                splits / f"{part}.csv", index=False
            )
    pd.DataFrame(rows, columns=INDEX_COLUMNS).to_csv(processed / f"{SOURCE}.index.csv", index=False)
    (processed / f"{SOURCE}.meta.json").write_text(
        json.dumps({"clips": row, "clip_samples": SAMPLES})
    )
    write_manifest(processed, manifest)
    return manifest
