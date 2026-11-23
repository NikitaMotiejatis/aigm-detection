"""Model cache, outbox, registry and run queue (WP3 steps 4, 5, 7), on a local MLflow store."""

import importlib.util
import sys
from pathlib import Path

import mlflow
import pytest

from aigm import tracking
from aigm.cli import main
from aigm.config import load_config

REPO = Path(__file__).resolve().parents[1]
CONFIG = str(REPO / "configs" / "smoke.yaml")
TESTSETS = str(REPO / "configs" / "testsets.yaml")


def load_submit():
    spec = importlib.util.spec_from_file_location("submit", REPO / "scripts" / "submit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["submit"] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("AIGM_GIT_COMMIT", "abc123")
    monkeypatch.setenv("AIGM_GIT_DIRTY", "false")
    mlflow.set_tracking_uri(uri)
    assert main(["train", "--config", CONFIG]) == 0
    return mlflow.search_runs(experiment_names=["smoke"], output_format="list")[0]


def eval_runs(run_id):
    return mlflow.search_runs(experiment_names=["smoke"], output_format="list",
                              filter_string=f"tags.eval_of = '{run_id}'")  # fmt: skip


def test_offline_eval_waits_in_outbox_and_sync_uploads_once(store, monkeypatch):
    run_id = store.info.run_id
    assert main(["register", "smoke", "--suffix", "test"]) == 0
    assert main(["pull"]) == 0
    assert (Path("models") / "smallcnn-test" / "1" / "best.pt").exists()

    with monkeypatch.context() as offline:
        offline.setattr(tracking, "server_reachable", lambda timeout=5.0: False)
        args = ["eval", "--model", "smallcnn-test", "--testset", "smoke_val",
                "--testsets", TESTSETS]  # fmt: skip
        assert main(args) == 0
        assert len(tracking.pending_entries()) == 1 and not eval_runs(run_id)
        assert main(["sync"]) == 1

    assert main(["sync"]) == 0
    runs = eval_runs(run_id)
    assert len(runs) == 1 and tracking.pending_entries() == []
    assert runs[0].data.metrics["eer"] == pytest.approx(store.data.metrics["best_val_eer"])
    assert runs[0].data.tags["registry_model"] == "smallcnn-test"


def test_upload_is_idempotent(store):
    entry = tracking.write_outbox("smoke", "again", {"eval_of": "x"}, {"a": 1}, {"m": 1.0}, {})
    copy = entry.with_name("copy")
    copy.mkdir()
    (copy / "result.json").write_text((entry / "result.json").read_text())
    first = tracking.upload_entry(entry)
    assert tracking.upload_entry(copy) == first
    assert len(eval_runs("x")) == 1


def test_register_keeps_version_when_best_run_is_unchanged(store):
    first = tracking.register_best("smoke", "test")
    assert tracking.register_best("smoke", "test") == first
    assert first["smallcnn-test"] == (store.info.run_id, "1")


def test_queue_skips_finished_clean_runs_only(store, capsys):
    submit = load_submit()
    sweeps = Path("sweeps.yaml")
    sweeps.write_text(f"smoke: {{config: {CONFIG}, seeds: [0], slots: 1, variants: [[]]}}\n")
    args = ["smoke", "--image", "aigm:test", "--dry-run", "--sweeps", str(sweeps)]
    assert submit.main(args) == 0
    out = capsys.readouterr().out
    assert "1 runs, 1 already finished" in out and "docker run" not in out

    client = mlflow.MlflowClient()
    client.set_tag(store.info.run_id, "dirty", "true")
    assert submit.main(args) == 0
    out = capsys.readouterr().out
    assert "1 runs, 0 already finished" in out and "docker run" in out


def test_config_hash_ignores_the_cache_location():
    a = load_config(REPO / "configs" / "e1_indomain.yaml", ["processed_dir=/a"])
    b = load_config(REPO / "configs" / "e1_indomain.yaml", ["processed_dir=/b"])
    c = load_config(REPO / "configs" / "e1_indomain.yaml", ["processed_dir=/a", "seed=1"])
    assert tracking.config_hash(a) == tracking.config_hash(b) != tracking.config_hash(c)
