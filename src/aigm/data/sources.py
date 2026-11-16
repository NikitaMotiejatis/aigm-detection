"""One reader per dataset. Each returns one row per raw file with the columns in COLUMNS.

`group_id` is the unit that must never be split between train, validation and test:
the YouTube id (FakeMusicCaps, MusicCaps), the artist id (FMA) or the original track (Echoes).
`path` is relative to the raw data directory.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

REAL, GENERATED = 0, 1
COLUMNS = ["item_id", "source", "label", "generator", "group_id", "path", "variant"]
FMC_GENERATORS = ["MusicGen_medium", "audioldm2", "musicldm", "mustango", "stable_audio_open"]


def _table(rows: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(rows, columns=COLUMNS).astype({"label": int})
    for column in COLUMNS:
        if column != "label":
            table[column] = table[column].astype(object)
    if table["item_id"].duplicated().any():
        raise ValueError("duplicate item ids")
    return table.sort_values("item_id", kind="stable").reset_index(drop=True)


def fakemusiccaps(raw: Path) -> pd.DataFrame:
    rows = []
    for generator in FMC_GENERATORS:
        for path in sorted((raw / "fakemusiccaps" / generator).glob("*.wav")):
            rows.append({
                "item_id": f"fmc/{generator}/{path.stem}", "source": "fakemusiccaps",
                "label": GENERATED, "generator": generator, "group_id": path.stem,
                "path": f"fakemusiccaps/{generator}/{path.name}", "variant": "",
            })  # fmt: skip
    return _table(rows)


def musiccaps(raw: Path) -> pd.DataFrame:
    rows = [
        {
            "item_id": f"musiccaps/{path.stem}", "source": "musiccaps", "label": REAL,
            "generator": "real", "group_id": path.stem, "path": f"musiccaps/{path.name}",
            "variant": "",
        }  # fmt: skip
        for path in sorted((raw / "musiccaps").glob("*.wav"))
    ]
    return _table(rows)


def fma_small(raw: Path) -> pd.DataFrame:
    """`variant` holds the top-level genre, which the split is stratified by."""
    tracks = pd.read_csv(raw / "fma_metadata" / "tracks.csv", header=[0, 1], index_col=0)
    rows = []
    for path in sorted((raw / "fma_small").glob("*/*.mp3")):
        track = int(path.stem)
        rows.append({
            "item_id": f"fma/{path.stem}", "source": "fma_small", "label": REAL,
            "generator": "real", "group_id": f"artist{int(tracks.at[track, ('artist', 'id')])}",
            "path": f"fma_small/{path.parent.name}/{path.name}",
            "variant": str(tracks.at[track, ("track", "genre_top")]),
        })  # fmt: skip
    return _table(rows)


def echoes(raw: Path) -> pd.DataFrame:
    """Rows come from the manifest only. `variant` is TTA (text-to-audio) or ATA."""
    root = raw / "echoes" / "Echoes"
    manifest = pd.read_csv(root / "dataset_manifest.csv").drop_duplicates("path_in_dataset")
    rows = [
        {
            "item_id": f"echoes/{Path(row.path_in_dataset).with_suffix('')}", "source": "echoes",
            "label": GENERATED, "generator": row.generator, "group_id": row.original_audio,
            "path": f"echoes/Echoes/{row.path_in_dataset}", "variant": row.type,
        }  # fmt: skip
        for row in manifest.itertuples()
    ]
    return _table(rows)


SOURCES: dict[str, Callable[[Path], pd.DataFrame]] = {
    "fakemusiccaps": fakemusiccaps,
    "musiccaps": musiccaps,
    "fma_small": fma_small,
    "echoes": echoes,
}
