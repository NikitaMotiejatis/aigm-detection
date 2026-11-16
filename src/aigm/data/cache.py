"""The processed cache: per source one raw int16 array, an index and a description.

    <source>.i16          int16, shape [clips, clip_samples], no header
    <source>.index.csv    one row per clip; `row` is the position in the array
    <source>.meta.json    shape and the conversion settings
    <source>.skipped.csv  raw files that could not be converted, with the reason

The cache is built once on machine A and copied to B. `write_manifest` and `verify` prove
that two copies are identical.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

INDEX_COLUMNS = [
    "row", "clip_id", "source", "label", "generator", "group_id", "track_id", "variant",
    "path", "offset_s", "valid_samples",
]  # fmt: skip
_TEXT_COLUMNS = ["clip_id", "source", "generator", "group_id", "track_id", "variant", "path"]


def read_index(processed: Path, source: str) -> pd.DataFrame:
    path = Path(processed) / f"{source}.index.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found: the processed cache for '{source}' is missing. It is built once "
            "on machine A with `aigm prepare` and copied to other machines, never rebuilt there."
        )
    return pd.read_csv(path, dtype=dict.fromkeys(_TEXT_COLUMNS, str), keep_default_na=False)


def read_meta(processed: Path, source: str) -> dict:
    return json.loads((Path(processed) / f"{source}.meta.json").read_text())


def open_array(processed: Path, source: str) -> np.memmap:
    meta = read_meta(processed, source)
    path = Path(processed) / f"{source}.i16"
    shape = (meta["clips"], meta["clip_samples"])
    if path.stat().st_size != shape[0] * shape[1] * 2:
        raise ValueError(f"{path} does not have the size its meta file states")
    return np.memmap(path, dtype=np.int16, mode="r", shape=shape)


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(processed: Path, manifest: Path) -> int:
    """Write `sha256  name` for every file in the cache, in the format of sha256sum."""
    files = sorted(p for p in Path(processed).iterdir() if p.is_file())
    if not files:
        raise FileNotFoundError(f"no files in {processed}")
    lines = [f"{sha256_file(p)}  {p.name}\n" for p in files]
    Path(manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest).write_text("".join(lines))
    return len(lines)


def read_manifest(manifest: Path) -> dict[str, str]:
    entries = {}
    for line in Path(manifest).read_text().splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    return entries


def verify(processed: Path, manifest: Path) -> list[str]:
    """Compare the cache with the committed manifest; return the list of problems."""
    expected = read_manifest(manifest)
    processed = Path(processed)
    present = {p.name for p in processed.iterdir() if p.is_file()} if processed.is_dir() else set()
    problems = [f"missing: {name}" for name in sorted(set(expected) - present)]
    problems += [f"not in the manifest: {name}" for name in sorted(present - set(expected))]
    for name in sorted(set(expected) & present):
        if sha256_file(processed / name) != expected[name]:
            problems.append(f"different content: {name}")
    return problems
