#!/usr/bin/env python
"""Run queue, machine A only (WP3 step 4): `make submit EXP=e1_indomain`.

Expands an entry of configs/sweeps.yaml into `docker run` commands. Runs that already finished
on a clean commit are skipped, so repeating the command resumes a sweep and no queue state is kept.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf

from aigm.config import load_config
from aigm.tracking import config_hash

REPO = Path(__file__).resolve().parents[1]


@dataclass
class Job:
    overrides: list[str]
    experiment: str
    config_sha256: str
    device: str
    synthetic: bool


def expand(sweep, extra: list[str]) -> list[Job]:
    jobs = []
    for variant in sweep.variants:
        for seed in sweep.seeds:
            overrides = [*variant, f"seed={seed}", *extra]
            cfg = load_config(REPO / sweep.config, overrides)
            jobs.append(Job(overrides, cfg.experiment, config_hash(cfg), cfg.device,
                            bool(cfg.data.get("synthetic", False))))  # fmt: skip
    return jobs


def is_done(client: MlflowClient, job: Job) -> bool:
    experiment = client.get_experiment_by_name(job.experiment)
    if experiment is None:
        return False
    query = (f"tags.config_sha256 = '{job.config_sha256}' and tags.kind = 'train' "
             "and tags.dirty = 'false' and attributes.status = 'FINISHED'")  # fmt: skip
    return bool(client.search_runs([experiment.experiment_id], query, max_results=1))


def docker_command(job: Job, config: str, image: str) -> list[str]:
    outputs = REPO / "outputs"
    outputs.mkdir(exist_ok=True)
    command = ["docker", "run", "--rm", "--network", "host", "--shm-size", "8g",
               "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp",
               "-e", "TORCH_HOME=/app/outputs/torch",
               "-e", f"MLFLOW_TRACKING_URI={os.environ['MLFLOW_TRACKING_URI']}",
               "-e", f"AIGM_MACHINE={os.environ.get('AIGM_MACHINE', os.uname().nodename)}",
               "-v", f"{outputs}:/app/outputs"]  # fmt: skip
    if job.device != "cpu":
        command += ["--gpus", "all"]
    if not job.synthetic:
        processed = Path(os.environ.get("PROCESSED_DIR", REPO / "data/processed")).resolve()
        command += ["-v", f"{processed}:/app/data/processed:ro"]
    return [*command, image, "train", "--config", config, *job.overrides]


def container_reaches_server(image: str) -> bool:
    """Check from inside the image: its newer OpenSSL sends larger TLS packets than host curl."""
    uri = os.environ["MLFLOW_TRACKING_URI"]
    if not uri.startswith("http"):
        return True
    probe = ("import os, urllib.request as r; "
             "uri = os.environ['MLFLOW_TRACKING_URI'].rstrip('/'); "
             "r.urlopen(uri + '/health', timeout=10)")  # fmt: skip
    command = ["docker", "run", "--rm", "--network", "host", "-e", f"MLFLOW_TRACKING_URI={uri}",
               "--entrypoint", "python", image, "-c", probe]  # fmt: skip
    try:
        done = subprocess.run(command, check=False, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        reason = "the check timed out"
    else:
        if done.returncode == 0:
            return True
        reason = (done.stderr.strip().splitlines() or ["docker run failed"])[-1]
    print(f"ERROR: the container cannot reach MLflow at {uri}: {reason}\n"
          "If this is a TLS handshake timeout, check the network MTU (README, section 1):\n"
          "  ip link show eth0  # must say mtu 1500\n"
          "  sudo ip link set dev eth0 mtu 1500", file=sys.stderr)  # fmt: skip
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--image", required=True)
    parser.add_argument("--slots", type=int, default=None, help="default: from the sweep file")
    parser.add_argument("--sweeps", default=str(REPO / "configs/sweeps.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("overrides", nargs="*", metavar="key=value")
    args = parser.parse_args(argv)

    if "MLFLOW_TRACKING_URI" not in os.environ:
        parser.error("MLFLOW_TRACKING_URI is not set")
    sweeps = OmegaConf.load(args.sweeps)
    if args.experiment not in sweeps:
        parser.error(f"no sweep '{args.experiment}'; have {sorted(sweeps)}")
    sweep = sweeps[args.experiment]

    client = MlflowClient()
    jobs = expand(sweep, args.overrides)
    pending = [job for job in jobs if not is_done(client, job)]
    print(f"{args.experiment}: {len(jobs)} runs, {len(jobs) - len(pending)} already finished",
          flush=True)  # fmt: skip
    commands = [docker_command(job, sweep.config, args.image) for job in pending]
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return 0
    if commands and not container_reaches_server(args.image):
        return 1

    def run(command: list[str]) -> int:
        return subprocess.run(command, check=False).returncode

    with ThreadPoolExecutor(args.slots or int(sweep.slots)) as pool:
        codes = list(pool.map(run, commands))
    failed = [job for job, code in zip(pending, codes, strict=True) if code != 0]
    for job in failed:
        print(f"FAILED: {' '.join(job.overrides)}", file=sys.stderr)
    print(f"{args.experiment}: {len(pending) - len(failed)} finished, {len(failed)} failed")

    if pending and args.experiment != "smoke" and not args.no_backup:
        if subprocess.run([str(REPO / "scripts/mlflow_backup.sh")], check=False).returncode:
            print("WARNING: the MLflow backup failed", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
