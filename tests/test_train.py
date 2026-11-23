from pathlib import Path

import mlflow
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from aigm.cli import main
from aigm.config import load_config
from aigm.models import BACKBONES, build_model, detection_score
from aigm.models.frontend import LogMel

REPO = Path(__file__).resolve().parents[1]


def test_frontend_shape_and_standardisation():
    x = LogMel()(torch.randn(3, 160_000) * 0.1)
    assert x.shape == (3, 1, 224, 224)
    assert x.mean(dim=(2, 3)).abs().max() < 1e-4
    assert (x.std(dim=(2, 3)) - 1).abs().max() < 1e-4


def test_frontend_is_gain_invariant_and_handles_silence():
    wave = torch.randn(1, 160_000) * 0.1
    frontend = LogMel()
    assert torch.allclose(frontend(wave), frontend(wave * 0.5), atol=1e-3)
    assert torch.isfinite(frontend(torch.zeros(1, 160_000))).all()


@pytest.mark.parametrize("name", sorted(BACKBONES))
def test_models_take_a_waveform_and_return_two_logits(name):
    cfg = OmegaConf.create({"model": {"name": name, "pretrained": False},
                            "frontend": {"n_mels": 64}})  # fmt: skip
    model = build_model(cfg).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 160_000) * 0.1)
    assert logits.shape == (2, 2)
    assert detection_score(logits).shape == (2,)


def test_smoke_train_then_eval_reproduces_validation_eer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    config = str(REPO / "configs" / "smoke.yaml")
    testsets = str(REPO / "configs" / "testsets.yaml")
    assert main(["train", "--config", config]) == 0

    train_run = mlflow.search_runs(experiment_names=["smoke"], output_format="list")[0]
    assert train_run.info.status == "FINISHED"
    for tag in ("git_commit", "dirty", "image_tag", "machine", "gpu", "cache_sha256",
                "split_sha256.train.csv", "split_sha256.val.csv"):  # fmt: skip
        assert tag in train_run.data.tags
    run_id = train_run.info.run_id
    scores = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="scores/val.csv")
    stored = pd.read_csv(scores)
    assert len(stored) == 12

    assert main(["eval", "--run", run_id, "--testset", "smoke_val", "--testsets", testsets]) == 0
    runs = mlflow.search_runs(experiment_names=["smoke"], output_format="list",
                              filter_string=f"tags.eval_of = '{run_id}'")  # fmt: skip
    assert len(runs) == 1
    assert runs[0].data.metrics["eer"] == pytest.approx(
        train_run.data.metrics["best_val_eer"], abs=1e-9
    )


def test_training_is_repeatable_on_cpu(tmp_path, monkeypatch):
    from aigm.train import train

    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    cfg = load_config(REPO / "configs" / "smoke.yaml")
    runs = [train(cfg), train(cfg)]
    losses = [mlflow.get_run(r).data.metrics["train_loss"] for r in runs]
    assert losses[0] == losses[1]
