"""Entry point of the `aigm` command.

Subcommands are registered here as they are implemented; nothing is listed before it works.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from aigm import __version__

DATA_CONFIG = "configs/data.yaml"
MANIFEST = "splits/CACHE.sha256"


def _config(args: argparse.Namespace):
    from aigm.config import load_config

    return load_config(args.config, args.overrides)


def _audit(args: argparse.Namespace) -> int:
    from aigm.data.audit import audit

    summary = audit(_config(args), Path(args.out), args.workers)
    print(summary.to_string(index=False))
    print(f"\nwritten: {args.out}/audit.csv, {args.out}/audit_spectra.pdf")
    return 0


def _prepare(args: argparse.Namespace) -> int:
    from aigm.data.prepare import prepare
    from aigm.data.sources import SOURCES

    cfg = _config(args)
    for source in SOURCES if args.source == "all" else [args.source]:
        meta = prepare(source, cfg, force=args.force, workers=args.workers)
        print(f"{source}: {meta['clips']} clips from {meta['files']} files, "
              f"{meta['skipped']} skipped")  # fmt: skip
    return 0


def _make_splits(args: argparse.Namespace) -> int:
    from aigm.data.splits import make_splits

    print(make_splits(_config(args), force=args.force))
    return 0


def _data_manifest(args: argparse.Namespace) -> int:
    from aigm.data.cache import write_manifest

    n = write_manifest(Path(_config(args).processed_dir), Path(args.manifest))
    print(f"{args.manifest}: {n} files")
    return 0


def _data_verify(args: argparse.Namespace) -> int:
    from aigm.data.cache import read_manifest, verify

    processed = Path(_config(args).processed_dir)
    problems = verify(processed, Path(args.manifest))
    for problem in problems:
        print(problem)
    if problems:
        print(f"FAILED: {processed} does not match {args.manifest}")
        return 1
    print(f"OK: {len(read_manifest(Path(args.manifest)))} files in {processed} match "
          f"{args.manifest}")  # fmt: skip
    return 0


def _train(args: argparse.Namespace) -> int:
    from aigm.train import train

    print(train(_config(args)))
    return 0


def _eval(args: argparse.Namespace) -> int:
    from aigm.evaluate import evaluate_run

    result = evaluate_run(args.run_id, args.testset, args.testsets, args.overrides)
    for name, value in result.items():
        print(f"{name}: {value:.6f}")
    return 0


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DATA_CONFIG, help=f"default: {DATA_CONFIG}")
    parser.add_argument("overrides", nargs="*", metavar="key=value", help="config overrides")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aigm",
        description="Experiments on detecting AI-generated music.",
    )
    parser.add_argument("--version", action="version", version=f"aigm {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("audit", help="statistics and mean spectra of the raw audio")
    p.add_argument("--out", default="results")
    p.add_argument("--workers", type=int, default=None)
    _add_config(p)
    p.set_defaults(run=_audit)

    p = sub.add_parser("prepare", help="convert a raw dataset into the processed cache")
    p.add_argument("source", choices=["fakemusiccaps", "musiccaps", "fma_small", "echoes", "all"])
    p.add_argument("--force", action="store_true", help="replace an existing cache")
    p.add_argument("--workers", type=int, default=None)
    _add_config(p)
    p.set_defaults(run=_prepare)

    p = sub.add_parser("make-splits", help="write the split files (run once, then committed)")
    p.add_argument("--force", action="store_true", help="replace existing split files")
    _add_config(p)
    p.set_defaults(run=_make_splits)

    data = sub.add_parser("data", help="checksums of the processed cache")
    data_sub = data.add_subparsers(dest="data_command", metavar="<command>", required=True)
    p = data_sub.add_parser("manifest", help="write the checksum manifest of the cache")
    p.add_argument("--manifest", default=MANIFEST)
    _add_config(p)
    p.set_defaults(run=_data_manifest)
    p = data_sub.add_parser("verify", help="compare the cache with the committed manifest")
    p.add_argument("--manifest", default=MANIFEST)
    _add_config(p)
    p.set_defaults(run=_data_verify)

    p = sub.add_parser("train", help="train one configuration and log it to MLflow")
    p.add_argument("--config", required=True)
    p.add_argument("overrides", nargs="*", metavar="key=value", help="config overrides")
    p.set_defaults(run=_train)

    p = sub.add_parser("eval", help="evaluate a run's checkpoint on a named test set")
    p.add_argument("--run", dest="run_id", required=True, help="MLflow run id of training")
    p.add_argument("--testset", required=True)
    p.add_argument("--testsets", default="configs/testsets.yaml")
    p.add_argument("overrides", nargs="*", metavar="key=value", help="config overrides")
    p.set_defaults(run=_eval)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
