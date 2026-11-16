import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from aigm.cli import main
from aigm.data.cache import open_array, read_index, read_meta, verify, write_manifest
from aigm.data.dataset import ClipDataset, balanced_sampler
from aigm.data.prepare import prepare
from aigm.data.sources import SOURCES
from aigm.data.splits import PARTS, bucket
from tests.conftest import REPO, YTIDS


def read_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"group_id": str}, keep_default_na=False)


def assert_groups_disjoint(splits: Path) -> None:
    for family in ("fmc", "fma"):
        groups = [set(read_split(splits / f"{family}_{p}.csv")["group_id"]) for p in PARTS]
        assert all(groups), f"an empty part in {family}"
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_sources(raw_dir):
    tables = {name: reader(raw_dir) for name, reader in SOURCES.items()}
    assert len(tables["fakemusiccaps"]) == 5 * len(YTIDS)
    assert set(tables["musiccaps"]["group_id"]) == set(YTIDS[:10])
    assert tables["fma_small"]["group_id"].nunique() == 18
    echoes = tables["echoes"]  # the duplicated manifest row and the unlisted file are dropped
    assert len(echoes) == 3 and not echoes["path"].str.contains("not_in_manifest").any()
    assert echoes["group_id"].nunique() == 2
    for table in tables.values():
        assert all((raw_dir / p).exists() for p in table["path"])


def test_cache_shape_and_index(cache):
    processed = Path(cache.processed_dir)
    expected = {"fakemusiccaps": 60, "musiccaps": 10, "fma_small": 19 * 3, "echoes": 6 + 2 + 1}
    for source, clips in expected.items():
        index, meta, array = (f(processed, source) for f in (read_index, read_meta, open_array))
        assert array.shape == (clips, 160000) and array.dtype == np.int16
        assert meta["audio"]["sample_rate"] == 16000 and len(index) == clips
        assert list(index["row"]) == list(range(clips)) and index["clip_id"].is_unique
        assert np.abs(array).max(axis=1).min() > 0
    skipped = pd.read_csv(processed / "fma_small.skipped.csv")
    assert list(skipped["path"]) == ["fma_small/000/000020.mp3"]
    echoes = read_index(processed, "echoes")
    short = echoes[echoes["path"].str.contains("song2")].iloc[0]
    assert short["valid_samples"] == 96000  # 6 s, zero-padded to 10 s
    assert not open_array(processed, "echoes")[short["row"], 100000:].any()


def test_prepare_refuses_to_rebuild(cache):
    with pytest.raises(FileExistsError):
        prepare("musiccaps", cache)


def test_missing_cache_is_an_error_not_a_rebuild(cache, tmp_path):
    with pytest.raises(FileNotFoundError, match="never rebuilt"):
        ClipDataset(Path(cache.splits.dir) / "fmc_train.csv", tmp_path)


def test_splits_keep_groups_together(cache):
    splits = Path(cache.splits.dir)
    assert_groups_disjoint(splits)
    fmc = pd.concat(read_split(splits / f"fmc_{p}.csv") for p in PARTS)
    assert set(fmc["group_id"]) == set(YTIDS[:10])  # ids without a real clip are left out
    assert (fmc.groupby("group_id")["label"].sum() == 5).all() and len(fmc) == 60
    fma = pd.concat(read_split(splits / f"fma_{p}.csv").assign(part=p) for p in PARTS)
    assert len(fma) == 57 and (fma.groupby("group_id")["part"].nunique() == 1).all()
    assert set(read_split(splits / "echoes_test.csv")["label"]) == {1}
    assert "sha256" in (splits / "SPLITS.md").read_text()


def test_bucket_is_stable_and_proportional():
    fractions = {"train": 0.8, "val": 0.1, "test": 0.1}
    parts = pd.Series([bucket(0, f"id{i}", fractions) for i in range(20000)])
    assert bucket(0, "id5", fractions) == parts[5]
    shares = parts.value_counts(normalize=True)
    assert abs(shares["train"] - 0.8) < 0.02 and abs(shares["val"] - 0.1) < 0.02


def test_dataset(cache):
    splits, processed = Path(cache.splits.dir), cache.processed_dir
    files = [splits / f"fmc_{p}.csv" for p in PARTS]
    full = ClipDataset(files, processed)
    assert len(full) == 60
    wave, label, clip_id, group_id = full[0]
    assert wave.shape == (160000,) and wave.dtype == torch.float32 and wave.abs().max() <= 1
    assert label in (0, 1) and group_id in clip_id
    one = ClipDataset(files, processed, generators=["musicldm"])
    assert len(one) == 20 and set(one.table["generator"]) == {"real", "musicldm"}
    assert len(ClipDataset(files, processed, sources=["musiccaps"])) == 10
    train = ClipDataset(files, processed, train=True)
    assert not torch.equal(train[0][0], train[0][0]) and torch.equal(full[0][0], full[0][0])
    sampler = balanced_sampler(full, torch.Generator().manual_seed(0))
    drawn = full.labels[list(sampler)]
    assert len(drawn) == 60 and 0.25 < drawn.mean() < 0.75
    loader = torch.utils.data.DataLoader(full, batch_size=4, num_workers=2)
    waves, labels, clip_ids, _ = next(iter(loader))
    assert waves.shape == (4, 160000) and len(clip_ids) == 4 and labels.dtype == torch.int64


def test_manifest_and_verify(cache, tmp_path):
    copy = tmp_path / "processed"
    shutil.copytree(cache.processed_dir, copy)
    manifest = tmp_path / "CACHE.sha256"
    assert write_manifest(copy, manifest) == 16
    args = ["--manifest", str(manifest), f"processed_dir={copy}"]
    assert verify(copy, manifest) == [] and main(["data", "verify", *args]) == 0
    with open(copy / "musiccaps.i16", "r+b") as f:
        f.seek(1000)
        f.write(b"\x01\x02")
    (copy / "echoes.meta.json").unlink()
    (copy / "stray.txt").write_text("x")
    assert verify(copy, manifest) == [
        "missing: echoes.meta.json", "not in the manifest: stray.txt",
        "different content: musiccaps.i16",
    ]  # fmt: skip
    assert main(["data", "verify", *args]) == 1


# The committed split files and, where it is present, the real cache.

COMMITTED = REPO / "splits"
has_splits = pytest.mark.skipif(not (COMMITTED / "fmc_train.csv").exists(),
                                reason="split files not generated yet")  # fmt: skip


@has_splits
def test_committed_splits_keep_groups_together():
    assert_groups_disjoint(COMMITTED)
    ids = set((COMMITTED / "musiccaps_ids.txt").read_text().split())
    fmc = pd.concat(read_split(COMMITTED / f"fmc_{p}.csv") for p in PARTS)
    assert set(fmc["group_id"]) <= ids and fmc["clip_id"].is_unique


@has_splits
def test_committed_splits_exist_in_cache():
    processed = REPO / "data" / "processed"
    if not (processed / "musiccaps.index.csv").exists():
        pytest.skip("processed cache not present on this machine")
    for name in sorted(COMMITTED.glob("*.csv")):
        split = read_split(name)
        for source, part in split.groupby("source"):
            index = read_index(processed, source)
            assert set(part["clip_id"]) <= set(index["clip_id"]), name.name
            assert open_array(processed, source).shape == (len(index), 160000)
