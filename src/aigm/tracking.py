"""What every run records in MLflow (WP2 step 6), model cache, outbox, registry (WP3 5, 7)."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import platform
import shutil
import subprocess
import tempfile
import urllib.request
import uuid
from pathlib import Path

import mlflow
import torch
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf

from aigm.data.cache import open_array, read_manifest, read_meta, sha256_file


def git_state() -> tuple[str, bool]:
    """Commit and dirty flag. In the image both were fixed by `make image`."""
    if commit := os.environ.get("AIGM_GIT_COMMIT"):
        return commit, os.environ.get("AIGM_GIT_DIRTY", "false") != "false"
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
    """Hash of what defines a run; the cache location differs between host and container."""
    values = OmegaConf.to_container(cfg, resolve=True)
    values.pop("processed_dir", None)
    text = OmegaConf.to_yaml(OmegaConf.create(values), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


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


def models_dir() -> Path:
    return Path(os.environ.get("AIGM_MODELS_DIR", "models"))


def outbox_dir() -> Path:
    return Path(os.environ.get("AIGM_OUTBOX_DIR", "outbox"))


def server_reachable(timeout: float = 5.0) -> bool:
    uri = mlflow.get_tracking_uri()
    if not uri.startswith("http"):
        return True
    try:
        with urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=timeout) as response:
            return response.status == 200
    except (OSError, http.client.HTTPException):
        return False


def _store_model(run_id: str, target: Path) -> Path:
    """Download a checkpoint; meta.json appears last, so a partial download never counts."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="model/best.pt", dst_path=tmp
        )
        checkpoint = torch.load(local, map_location="cpu", weights_only=True)
        target.mkdir(exist_ok=True)
        shutil.move(local, target / "best.pt")
    (target / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(checkpoint["config"])))
    meta = {"run_id": run_id, "val_threshold": checkpoint["val_threshold"]}
    (target / "meta.json").write_text(json.dumps(meta))
    return target / "best.pt"


def pull() -> list[str]:
    """Cache the current version of every registry model. Does nothing when offline."""
    if not server_reachable():
        return []
    client = MlflowClient()
    pulled = []
    for model in client.search_registered_models():
        versions = client.search_model_versions(f"name = '{model.name}'")
        latest = max(versions, key=lambda v: int(v.version), default=None)
        if latest is None or not latest.run_id:
            continue
        target = models_dir() / model.name / str(latest.version)
        if not (target / "meta.json").exists():
            _store_model(latest.run_id, target)
            pulled.append(f"{model.name}/{latest.version}")
    return pulled


def cached_model(name: str) -> tuple[Path, str]:
    """`name` or `name/version` from the local cache: (checkpoint, training run id)."""
    base, _, version = name.partition("/")
    root = models_dir() / base
    done = [p.parent.name for p in root.glob("*/meta.json")]
    if version:
        done = [v for v in done if v == version]
    if not done:
        raise FileNotFoundError(f"model '{name}' is not in {models_dir()}; run `aigm pull`")
    folder = root / max(done, key=int)
    return folder / "best.pt", json.loads((folder / "meta.json").read_text())["run_id"]


def run_checkpoint(run_id: str) -> Path:
    """Checkpoint of a run, from the cache if any copy is there, otherwise downloaded into it."""
    for meta in models_dir().glob("*/*/meta.json"):
        if json.loads(meta.read_text())["run_id"] == run_id:
            return meta.parent / "best.pt"
    return _store_model(run_id, models_dir() / "runs" / run_id)


def write_outbox(experiment: str, run_name: str, tags: dict, params: dict, metrics: dict,
                 artifacts: dict[str, Path]) -> Path:  # fmt: skip
    """Store a complete result; the final rename means an entry is never half written."""
    outbox_id = uuid.uuid4().hex
    root = outbox_dir()
    partial = root / f".{outbox_id}"
    for folder, path in artifacts.items():
        (partial / "artifacts" / folder).mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, partial / "artifacts" / folder / Path(path).name)
    partial.mkdir(parents=True, exist_ok=True)
    result = {"experiment": experiment, "run_name": run_name, "params": params,
              "tags": {**tags, "outbox_id": outbox_id},
              "metrics": {k: float(v) for k, v in metrics.items()}}  # fmt: skip
    (partial / "result.json").write_text(json.dumps(result))
    entry = root / outbox_id
    partial.rename(entry)
    return entry


def upload_entry(entry: Path) -> str:
    """Upload one outbox entry and remove it. Safe to repeat after an interrupted upload."""
    result = json.loads((entry / "result.json").read_text())
    experiment = mlflow.set_experiment(result["experiment"])
    client = MlflowClient()
    query = f"tags.outbox_id = '{result['tags']['outbox_id']}'"
    for run in client.search_runs([experiment.experiment_id], query):
        if run.info.status == "FINISHED":
            shutil.rmtree(entry)
            return run.info.run_id
        client.delete_run(run.info.run_id)  # left by an interrupted upload of this same entry
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=result["run_name"],
                          tags=result["tags"]) as run:  # fmt: skip
        mlflow.log_params(result["params"])
        log_finite(result["metrics"])
        if (entry / "artifacts").exists():
            mlflow.log_artifacts(str(entry / "artifacts"))
    shutil.rmtree(entry)
    return run.info.run_id


def pending_entries() -> list[Path]:
    root = outbox_dir()
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if (p / "result.json").exists()
                  and not p.name.startswith("."))  # fmt: skip


def sync() -> tuple[int, int]:
    """Upload everything pending; returns (uploaded, still pending)."""
    entries = pending_entries()
    if not entries or not server_reachable():
        return 0, len(entries)
    uploaded = 0
    for entry in entries:
        try:
            upload_entry(entry)
            uploaded += 1
        except (MlflowException, OSError) as error:
            print(f"upload of {entry.name} failed: {error}", flush=True)
    return uploaded, len(entries) - uploaded


def register_best(experiment: str, suffix: str) -> dict[str, tuple[str, str]]:
    """Register the best clean run per model as `<model>-<suffix>`, by validation EER."""
    client = MlflowClient()
    found = client.get_experiment_by_name(experiment)
    if found is None:
        raise ValueError(f"no experiment '{experiment}'")
    query = "tags.kind = 'train' and tags.dirty = 'false' and attributes.status = 'FINISHED'"
    best: dict[str, object] = {}
    for run in client.search_runs([found.experiment_id], query):
        if "best_val_eer" not in run.data.metrics:
            continue
        family = run.data.params["model.name"]
        key = (run.data.metrics["best_val_eer"], run.info.start_time)
        current = best.get(family)
        if current is None or key < (current.data.metrics["best_val_eer"],
                                     current.info.start_time):  # fmt: skip
            best[family] = run
    registered = {}
    for family, run in sorted(best.items()):
        name = f"{family}-{suffix}"
        if not client.search_registered_models(f"name = '{name}'"):
            client.create_registered_model(name)
        versions = client.search_model_versions(f"name = '{name}'")
        latest = max(versions, key=lambda v: int(v.version), default=None)
        if latest is None or latest.run_id != run.info.run_id:
            latest = client.create_model_version(name, f"runs:/{run.info.run_id}/model",
                                                 run_id=run.info.run_id)  # fmt: skip
        registered[name] = (run.info.run_id, str(latest.version))
    return registered
