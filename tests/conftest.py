"""A miniature copy of the raw data layout, filled with synthetic audio."""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from aigm.audio.io import ffmpeg_exe
from aigm.config import load_config
from aigm.data.sources import FMC_GENERATORS

REPO = Path(__file__).resolve().parents[1]
YTIDS = [f"yt{i:02d}_-AbCdE" for i in range(12)]


def tone(seconds: float, rate: int, seed: int, channels: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    x = 0.3 * np.sin(2 * np.pi * rng.uniform(100, 3000) * t) + 0.05 * rng.standard_normal(len(t))
    return np.stack([x] * channels, axis=1) if channels > 1 else x


def write_mp3(path: Path, x: np.ndarray, rate: int) -> None:
    wav = path.with_suffix(".tmp.wav")
    sf.write(wav, x, rate, subtype="PCM_16")
    subprocess.run([ffmpeg_exe(), "-v", "error", "-y", "-i", str(wav), "-b:a", "128k", str(path)],
                   check=True)  # fmt: skip
    wav.unlink()


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory) -> Path:
    raw = tmp_path_factory.mktemp("raw")
    seed = 0
    for generator in FMC_GENERATORS:
        (raw / "fakemusiccaps" / generator).mkdir(parents=True)
        for ytid in YTIDS:
            seed += 1
            sf.write(raw / "fakemusiccaps" / generator / f"{ytid}.wav",
                     tone(10.2, 16000, seed), 16000, subtype="FLOAT")  # fmt: skip
    (raw / "musiccaps").mkdir()
    for ytid in YTIDS[:10]:  # two ids have no real clip, as with removed videos
        seed += 1
        sf.write(raw / "musiccaps" / f"{ytid}.wav", tone(10, 44100, seed, 2), 44100,
                 subtype="PCM_16")  # fmt: skip

    (raw / "fma_small" / "000").mkdir(parents=True)
    (raw / "fma_metadata").mkdir()
    tracks = []
    for track in range(1, 21):
        seed += 1
        seconds = 12.0 if track == 20 else 30.0  # one track is too short and must be skipped
        write_mp3(raw / "fma_small" / "000" / f"{track:06d}.mp3", tone(seconds, 44100, seed), 44100)
        artist = track - 1 if track in (2, 12) else track  # two artists have two tracks
        tracks.append((track, artist, "Rock" if track <= 10 else "Folk"))
    frame = pd.DataFrame(
        {("artist", "id"): [t[1] for t in tracks], ("track", "genre_top"): [t[2] for t in tracks]},
        index=pd.Index([t[0] for t in tracks]),
    )
    frame.to_csv(raw / "fma_metadata" / "tracks.csv")

    root = raw / "echoes" / "Echoes"
    manifest = []
    for i, (generator, kind, seconds) in enumerate(
        [("suno", "TTA", 75.0), ("suno", "TTA", 25.0), ("musicgen", "ATA", 6.0)]
    ):
        seed += 1
        ext = "wav" if generator == "musicgen" else "mp3"
        rel = f"{kind}/{generator}/song{i}_{generator}_{kind}_001.{ext}"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        if ext == "wav":
            sf.write(root / rel, tone(seconds, 32000, seed), 32000)
        else:
            write_mp3(root / rel, tone(seconds, 48000, seed, 2), 48000)
        manifest.append({"path_in_dataset": rel, "original_audio": f"Song, {i // 2}",
                         "generator": generator, "type": kind, "genre": "Rock",
                         "description": "x", "duration": seconds})  # fmt: skip
    sf.write(root / "TTA" / "suno" / "not_in_manifest.wav", tone(20, 16000, 999), 16000)
    pd.DataFrame(manifest + manifest[:1]).to_csv(root / "dataset_manifest.csv", index=False)
    return raw


@pytest.fixture(scope="session")
def cfg(raw_dir, tmp_path_factory):
    work = tmp_path_factory.mktemp("work")
    (work / "splits").mkdir()
    (work / "splits" / "musiccaps_ids.txt").write_text("\n".join(YTIDS[:10]) + "\n")
    return load_config(REPO / "configs" / "data.yaml", [
        f"raw_dir={raw_dir}", f"processed_dir={work / 'processed'}",
        f"splits.dir={work / 'splits'}",
    ])  # fmt: skip


@pytest.fixture(scope="session")
def cache(cfg):
    """The processed cache and the split files built from the synthetic raw data."""
    from aigm.data.prepare import prepare
    from aigm.data.sources import SOURCES
    from aigm.data.splits import make_splits

    for source in SOURCES:
        prepare(source, cfg, workers=2)
    make_splits(cfg)
    return cfg
