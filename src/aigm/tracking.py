"""What every run records in MLflow (WP2 step 6)."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from pathlib import Path

import mlflow
import torch
from omegaconf import DictConfig, OmegaConf

from aigm.data.cache import open_array, read_manifest, read_meta, sha256_file


def git_state() -> tuple[str, bool]:
    """Commit and dirty flag. Without git (in the image) AIGM_GIT_COMMIT names a clean commit."""
    if commit := os.environ.get("AIGM_GIT_COMMIT"):
        return commit, False
    repo = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()  # fmt: skip
        status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"],
                                         text=True, stderr=subprocess.DEVNULL)  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True  # an unidentified state must not reach the report
    return commit, bool(status.strip())


def check_cache(processed: Path, manifest: Path, sources: list[str]) -> str:
    """Compare indexes and meta files with the manifest, check array sizes, return a hash.

    Hashing the 27 GB of audio on every run is too slow; `aigm data verify` does that.
    """
    entries = read_manifest(manifest)
    lines = []
    for source in sorted(sources):
        for suffix in ("i16", "index.csv", "meta.json"):
            name = f"{source}.{suffix}"
            if name not in entries:
                raise ValueError(f"{name} is not in {manifest}")
            if suffix != "i16" and sha256_file(Path(processed) / name) != entries[name]:
                raise ValueError(f"{name} differs from {manifest}; run `aigm data verify`")
            lines.append(f"{entries[name]}  {name}\n")
        read_meta(processed, source)
        open_array(processed, source)
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def flatten(value: dict, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, item in value.items():
        name = f"{prefix}{key}"
        if isinstance(item, dict):
            flat.update(flatten(item, f"{name}."))
        else:
            flat[name] = item
    return flat


def config_hash(cfg: DictConfig) -> str:
    return hashlib.sha256(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True).encode()).hexdigest()


def run_tags(kind: str, device: torch.device, cache_hash: str, splits: list[Path]) -> dict:
    commit, dirty = git_state()
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    tags = {
        "kind": kind,
        "git_commit": commit,
        "dirty": str(dirty).lower(),
        "image_tag": os.environ.get("AIGM_IMAGE_TAG", "none"),
        "machine": os.environ.get("AIGM_MACHINE", platform.node()),
        "gpu": gpu,
        "cache_sha256": cache_hash,
    }
    tags.update({f"split_sha256.{Path(s).name}": sha256_file(Path(s)) for s in splits})
    return tags


def log_finite(metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics, leaving out undefined ones (for example EER on a one-class test set)."""
    mlflow.log_metrics({k: v for k, v in metrics.items() if v == v}, step=step)
