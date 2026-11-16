"""Split files (WP1 step 4). Built once from the cache index, then committed; training and
evaluation only ever read them.

Every split is made at the level of `group_id`, so a real MusicCaps clip and the five clips
generated from its caption, or two tracks of one artist, always land on the same side.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from aigm.data.cache import read_index, read_meta, sha256_file
from aigm.data.sources import FMC_GENERATORS

SPLIT_COLUMNS = ["clip_id", "source", "label", "generator", "group_id"]
PARTS = ["train", "val", "test"]


def _hash(seed: int, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big")


def bucket(seed: int, group_id: str, fractions: dict[str, float]) -> str:
    """Part of a group from a seeded hash of its id. Adding or removing other groups never
    moves a group to a different part."""
    point = _hash(seed, group_id) % 10_000 / 10_000
    edge = 0.0
    for part in PARTS:
        edge += fractions[part]
        if point < edge:
            return part
    return PARTS[-1]


def split_fmc(fmc: pd.DataFrame, real: pd.DataFrame, ids: list[str], seed: int,
              fractions: dict[str, float]) -> dict[str, pd.DataFrame]:  # fmt: skip
    """FakeMusicCaps + MusicCaps: only ids with a real clip and all five generated clips."""
    complete = fmc.groupby("group_id")["generator"].nunique()
    complete = set(complete[complete == len(FMC_GENERATORS)].index)
    usable = set(ids) & set(real["group_id"]) & complete
    table = pd.concat([real, fmc])
    table = table[table["group_id"].isin(usable)]
    part = table["group_id"].map(lambda g: bucket(seed, g, fractions))
    return {p: table[part == p] for p in PARTS}


def split_by_group_stratified(table: pd.DataFrame, seed: int,
                              fractions: dict[str, float]) -> dict[str, pd.DataFrame]:  # fmt: skip
    """FMA: by artist, stratified by genre (`variant`).

    An artist counts towards the genre most of its clips have. Within a genre, artists are
    taken from the largest down (ties in hash order) and each goes to the part that is
    furthest below its target share of clips.
    """
    counts = table.groupby(["group_id", "variant"]).size().reset_index(name="n")
    counts = counts.sort_values(["group_id", "n", "variant"], ascending=[True, False, True])
    stratum = counts.drop_duplicates("group_id").set_index("group_id")["variant"]
    size = table.groupby("group_id").size()
    assignment = {}
    for _, groups in stratum.groupby(stratum):
        order = sorted(groups.index, key=lambda g: (-size[g], _hash(seed, g)))
        total = sum(size[g] for g in order)
        filled = dict.fromkeys(PARTS, 0)
        for g in order:
            part = max(PARTS, key=lambda p: fractions[p] - filled[p] / total)
            assignment[g] = part
            filled[part] += size[g]
    part = table["group_id"].map(assignment)
    return {p: table[part == p] for p in PARTS}


def _short(commit: str) -> str:
    return commit[:7] + ("-dirty" if commit.endswith("-dirty") else "")


def _markdown(table: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, table.columns)) + " |", "|" + "---|" * table.shape[1]]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in table.itertuples(index=False)]
    return "\n".join(lines)


def make_splits(cfg: DictConfig, force: bool = False) -> str:
    processed, out = Path(cfg.processed_dir), Path(cfg.splits.dir)
    seed, fractions = int(cfg.splits.seed), dict(cfg.splits.fractions)
    ids = (out / "musiccaps_ids.txt").read_text().split()
    index = {s: read_index(processed, s)
             for s in ("fakemusiccaps", "musiccaps", "fma_small", "echoes")}  # fmt: skip
    files: dict[str, pd.DataFrame] = {}
    for part, table in split_fmc(index["fakemusiccaps"], index["musiccaps"], ids, seed,
                                 fractions).items():  # fmt: skip
        files[f"fmc_{part}.csv"] = table
    for part, table in split_by_group_stratified(index["fma_small"], seed, fractions).items():
        files[f"fma_{part}.csv"] = table
    files["echoes_test.csv"] = index["echoes"]

    existing = [name for name in files if (out / name).exists()]
    if existing and not force:
        raise FileExistsError(
            f"{out / existing[0]} exists. Splits are frozen once committed; pass --force only "
            "if every run made on the old splits is to be discarded."
        )
    rows = []
    for name, table in files.items():
        table = table.sort_values("clip_id", kind="stable")
        table[SPLIT_COLUMNS].to_csv(out / name, index=False, lineterminator="\n")
        for (source, label, generator), g in table.groupby(["source", "label", "generator"]):
            rows.append({
                "file": name, "source": source, "class": "generated" if label else "real",
                "generator": generator, "clips": len(g), "tracks": g["track_id"].nunique(),
                "groups": g["group_id"].nunique(),
            })  # fmt: skip
    hashes = pd.DataFrame(
        [{"file": name, "clips": len(t), "groups": t["group_id"].nunique(),
          "sha256": sha256_file(out / name)} for name, t in files.items()]
    )  # fmt: skip
    cache = pd.DataFrame([
        {"source": s, "raw files": m["files"], "skipped": m["skipped"], "clips": m["clips"],
         "RMS normalisation": m["audio"]["normalise_rms"],
         "built at commit": _short(m["git_commit"])}
        for s, m in ((s, read_meta(processed, s)) for s in index)
    ])  # fmt: skip
    text = f"""# Splits

Generated by `aigm make-splits`; do not edit. Seed {seed}, target shares
{", ".join(f"{p} {fractions[p]:.0%}" for p in PARTS)}.

- `fmc_*`: MusicCaps (real) and FakeMusicCaps (generated). Only the {len(ids)} YouTube ids in
  `musiccaps_ids.txt` are used; the part of an id comes from a seeded hash of the id, so a real
  clip and the five clips generated from its caption are always in the same part.
- `fma_*`: FMA small, split by artist and stratified by top-level genre; three 10 s clips per
  track.
- `echoes_test.csv`: every Echoes track in the dataset manifest, up to six 10 s clips per
  track. Test only. `group_id` is the original track the generation was conditioned on.

A group (YouTube id, artist, original track) never appears in two parts of the same dataset.

## Files

{_markdown(hashes)}

## Clips per file, class and generator

{_markdown(pd.DataFrame(rows))}

## Processed cache the splits were built from

{_markdown(cache)}
"""
    (out / "SPLITS.md").write_text(text)
    return hashes.drop(columns="sha256").to_string(index=False)
