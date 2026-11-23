"""Dataset over the processed cache, selected by split files (WP1 step 5)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from aigm.data.cache import open_array, read_index


class ClipDataset(Dataset):
    """Returns (waveform float32 [samples] in [-1, 1), label, clip id, group id).

    split_files: one or more split CSV files; their rows are concatenated.
    sources:     keep only these sources (default: all in the split files).
    generators:  keep only generated clips of these generators; real clips are always kept.
    train:       apply the training augmentation, a random gain of +-gain_db.
    """

    def __init__(
        self,
        split_files: str | Path | Sequence[str | Path],
        processed_dir: str | Path,
        sources: Sequence[str] | None = None,
        generators: Sequence[str] | None = None,
        train: bool = False,
        gain_db: float = 6.0,
    ) -> None:
        if isinstance(split_files, str | Path):
            split_files = [split_files]
        table = pd.concat(
            [pd.read_csv(f, dtype={"clip_id": str, "group_id": str}, keep_default_na=False)
             for f in split_files],
            ignore_index=True,
        )  # fmt: skip
        if sources is not None:
            table = table[table["source"].isin(list(sources))]
        if generators is not None:
            table = table[(table["label"] == 0) | table["generator"].isin(list(generators))]
        if table.empty:
            raise ValueError("no clips left after filtering by source and generator")
        if table["clip_id"].duplicated().any():
            raise ValueError("a clip appears more than once in the split files")
        self.processed_dir = Path(processed_dir)
        rows = []
        for source, part in table.groupby("source", sort=False):
            index = read_index(self.processed_dir, source).set_index("clip_id")
            missing = part.loc[~part["clip_id"].isin(index.index), "clip_id"]
            if len(missing):
                raise KeyError(f"{len(missing)} clips of '{source}' are in the split but not "
                               f"in the cache, e.g. {missing.iloc[0]}")  # fmt: skip
            found = index.loc[part["clip_id"]]
            rows.append(part.assign(row=found["row"].to_numpy(),
                                    track_id=found["track_id"].to_numpy()))  # fmt: skip
        self.table = pd.concat(rows).reset_index(drop=True)
        self.labels = self.table["label"].to_numpy()
        self.train, self.gain_db = train, gain_db
        self._arrays: dict[str, np.memmap] = {}

    def __len__(self) -> int:
        return len(self.table)

    def _array(self, source: str) -> np.memmap:
        # opened on first use, so that every data loader worker has its own handle
        if source not in self._arrays:
            self._arrays[source] = open_array(self.processed_dir, source)
        return self._arrays[source]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, str, str]:
        item = self.table.iloc[i]
        samples = np.array(self._array(item["source"])[item["row"]], dtype=np.float32)
        wave = torch.from_numpy(samples) / 32768.0
        if self.train and self.gain_db > 0:
            gain = torch.empty(1).uniform_(-self.gain_db, self.gain_db)
            wave = wave * 10 ** (gain / 20)
        return wave, int(item["label"]), item["clip_id"], item["group_id"]

    def __getstate__(self) -> dict:
        return {**self.__dict__, "_arrays": {}}


def balanced_sampler(
    dataset: ClipDataset, generator: torch.Generator | None = None
) -> WeightedRandomSampler:
    """Draws real and generated clips with equal probability (E1 has five generated per real)."""
    counts = np.bincount(dataset.labels, minlength=2)
    weights = 1.0 / counts[dataset.labels]
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), len(dataset), replacement=True,
        generator=generator,
    )  # fmt: skip
