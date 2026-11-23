"""Training loop (WP2 step 3)."""

from __future__ import annotations

import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from aigm import tracking
from aigm.data.dataset import ClipDataset, balanced_sampler
from aigm.data.synthetic import ensure_cache
from aigm.metrics import average_per_track, summarise
from aigm.models import build_model, detection_score


@dataclass
class DataPaths:
    root: Path
    processed: Path
    manifest: Path

    def split(self, name: str) -> Path:
        return self.root / name


def data_paths(cfg: DictConfig) -> DataPaths:
    root = Path(cfg.data.root)
    if cfg.data.get("synthetic", False):
        return DataPaths(root, root / "processed", ensure_cache(root))
    return DataPaths(root, Path(cfg.processed_dir), Path(cfg.manifest))


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_worker(_: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, batch_size: int, workers: int, device: torch.device, seed: int,
                sampler=None) -> DataLoader:  # fmt: skip
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
        worker_init_fn=_seed_worker, generator=generator,
        drop_last=sampler is not None and len(dataset) % batch_size == 1,  # BatchNorm needs 2
    )  # fmt: skip


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> np.ndarray:
    model.eval()
    scores = []
    for wave, *_ in loader:
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            logits = model(wave.to(device, non_blocking=True))
        scores.append(detection_score(logits).float().cpu().numpy())
    return np.concatenate(scores).astype(np.float64)


def clip_scores(dataset: ClipDataset, scores: np.ndarray):
    columns = ["clip_id", "track_id", "group_id", "generator", "label"]
    return dataset.table[columns].assign(score=scores)


def train(cfg: DictConfig) -> str:
    """Train one configuration and return its MLflow run id."""
    seed_everything(int(cfg.seed))
    device = select_device(cfg.device)
    paths = data_paths(cfg)
    sources = list(cfg.data.sources)
    generators = cfg.data.get("train_generators")
    generators = list(generators) if generators is not None else None
    train_files = [paths.split(f) for f in cfg.data.train]
    val_files = [paths.split(f) for f in cfg.data.val]
    cache_hash = tracking.check_cache(paths.processed, paths.manifest, sources)

    train_set = ClipDataset(train_files, paths.processed, sources, generators, train=True,
                            gain_db=float(cfg.data.gain_db))  # fmt: skip
    val_set = ClipDataset(val_files, paths.processed, sources, generators)
    sampler = balanced_sampler(train_set, torch.Generator().manual_seed(int(cfg.seed)))
    t = cfg.train
    workers = int(cfg.data.num_workers)
    train_loader = make_loader(train_set, t.batch_size, workers, device, cfg.seed, sampler)
    val_loader = make_loader(val_set, t.eval_batch_size, workers, device, cfg.seed)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(t.epochs) * len(train_loader)
    )
    use_amp = bool(t.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss()

    mlflow.set_experiment(cfg.experiment)
    tags = tracking.run_tags("train", device, cache_hash, train_files + val_files)
    tags["config_sha256"] = tracking.config_hash(cfg)
    tags["experiment_name"] = cfg.experiment
    with mlflow.start_run(run_name=f"{cfg.model.name}-seed{cfg.seed}", tags=tags) as run, \
            tempfile.TemporaryDirectory() as tmp:  # fmt: skip
        container = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_params(tracking.flatten(container))
        mlflow.log_text(OmegaConf.to_yaml(cfg, resolve=True), "config.yaml")
        checkpoint = Path(tmp) / "best.pt"
        best_key, best, stale = (float("inf"), float("inf")), None, 0

        for epoch in range(1, int(t.epochs) + 1):
            started = time.monotonic()
            model.train()
            total = 0.0
            for wave, labels, *_ in train_loader:
                wave = wave.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=use_amp):
                    loss = criterion(model(wave), labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"loss is {loss.item()} in epoch {epoch}")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total += loss.item()

            frame = clip_scores(val_set, predict(model, val_loader, device, use_amp))
            tracks = average_per_track(frame)
            val = summarise(tracks["label"], tracks["score"])
            logits = torch.tensor(tracks["score"].to_numpy())
            targets = torch.tensor(tracks["label"].to_numpy(), dtype=torch.float64)
            val_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets).item()
            tracking.log_finite(
                {"train_loss": total / len(train_loader), "val_loss": val_loss,
                 "val_eer": val["eer"], "val_auc": val["auc"],
                 "lr": scheduler.get_last_lr()[0],
                 "epoch_seconds": time.monotonic() - started}, step=epoch,
            )  # fmt: skip
            print(f"epoch {epoch}/{t.epochs}: train loss {total / len(train_loader):.4f}, "
                  f"val EER {val['eer']:.4f}, val AUC {val['auc']:.4f}", flush=True)  # fmt: skip

            key = (val["eer"], val_loss)  # many epochs can reach EER 0; the loss breaks the tie
            if key < best_key:
                best_key, best, stale = key, {"epoch": epoch, "frame": frame, "val": val}, 0
                torch.save({"state_dict": model.state_dict(), "config": container, "epoch": epoch,
                            "val_eer": val["eer"], "val_threshold": val["eer_threshold"]},
                           checkpoint)  # fmt: skip
            else:
                stale += 1
                if stale >= int(t.patience):
                    print(f"early stop after epoch {epoch}", flush=True)
                    break
        if best is None:
            raise RuntimeError("no epoch produced a usable validation EER")

        tracking.log_finite({f"best_val_{k}": v for k, v in best["val"].items()})
        mlflow.log_metric("best_epoch", best["epoch"])
        scores = Path(tmp) / "val.csv"
        best["frame"].to_csv(scores, index=False)
        mlflow.log_artifact(str(scores), "scores")
        mlflow.log_artifact(str(checkpoint), "model")
        print(f"run {run.info.run_id}: best epoch {best['epoch']}, "
              f"val EER {best['val']['eer']:.4f}", flush=True)  # fmt: skip
        return run.info.run_id
