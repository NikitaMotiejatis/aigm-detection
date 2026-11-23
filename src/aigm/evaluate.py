"""Evaluate a stored checkpoint on a named test set (WP2 step 4)."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from aigm import tracking
from aigm.data.dataset import ClipDataset
from aigm.metrics import average_per_track, bootstrap_ci, eer, summarise
from aigm.models import build_model
from aigm.train import clip_scores, data_paths, make_loader, predict, select_device


def metrics_for(clips: pd.DataFrame, threshold: float, n_boot: int, seed: int) -> dict[str, float]:
    """Overall and per-generator metrics on track-averaged scores."""
    tracks = average_per_track(clips)
    labels, scores = tracks["label"].to_numpy(), tracks["score"].to_numpy()
    result = summarise(labels, scores, threshold)
    low, high = bootstrap_ci(labels, scores, tracks["group_id"], lambda y, s: eer(y, s)[0],
                             n_boot=n_boot, seed=seed)  # fmt: skip
    result |= {"eer_ci_low": low, "eer_ci_high": high}
    real = tracks[tracks["label"] == 0]
    for generator, part in tracks[tracks["label"] == 1].groupby("generator"):
        both = pd.concat([real, part])
        one = summarise(both["label"], both["score"], threshold)
        result |= {f"{generator}/{k}": one[k] for k in ("eer", "auc", "tpr")}
    return result


def evaluate_run(
    run_id: str,
    testset: str,
    testsets_file: str | Path = "configs/testsets.yaml",
    overrides: Sequence[str] = (),
) -> dict[str, float]:
    local = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model/best.pt")
    checkpoint = torch.load(local, map_location="cpu", weights_only=True)
    cfg = OmegaConf.create(checkpoint["config"])
    if not cfg.data.get("synthetic", False):  # the cache location belongs to this machine
        cfg.processed_dir = os.environ.get("PROCESSED_DIR", "data/processed")
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    testsets = OmegaConf.load(testsets_file)
    if testset not in testsets:
        raise KeyError(f"no test set '{testset}' in {testsets_file}; have {sorted(testsets)}")
    entry = testsets[testset]
    if entry.get("perturbation") is not None:
        raise NotImplementedError("perturbed test sets arrive with `aigm perturb` (E5, E7)")

    paths = data_paths(cfg)
    files = [paths.split(f) for f in entry.splits]
    sources = list(entry.sources)
    generators = list(entry.generators) if entry.get("generators") is not None else None
    cache_hash = tracking.check_cache(paths.processed, paths.manifest, sources)
    dataset = ClipDataset(files, paths.processed, sources, generators)

    device = select_device(cfg.device)
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    loader = make_loader(dataset, cfg.train.eval_batch_size, int(cfg.data.num_workers), device,
                         int(cfg.seed))  # fmt: skip
    clips = clip_scores(dataset, predict(model, loader, device, bool(cfg.train.amp)))
    threshold = float(checkpoint["val_threshold"])
    result = metrics_for(clips, threshold, int(cfg.eval.n_boot), int(cfg.seed))

    mlflow.set_experiment(cfg.experiment)
    tags = tracking.run_tags("eval", device, cache_hash, files)
    tags |= {"eval_of": run_id, "testset": testset, "experiment_name": cfg.experiment}
    with mlflow.start_run(run_name=f"eval-{testset}-{run_id[:8]}", tags=tags), \
            tempfile.TemporaryDirectory() as tmp:  # fmt: skip
        mlflow.log_params({"testset": testset, "source_run": run_id, "val_threshold": threshold,
                           "checkpoint_epoch": checkpoint["epoch"]})  # fmt: skip
        tracking.log_finite({k: v for k, v in result.items() if np.isscalar(v)})
        scores = Path(tmp) / f"{testset}.csv"
        clips.to_csv(scores, index=False)
        mlflow.log_artifact(str(scores), "scores")
    return result
