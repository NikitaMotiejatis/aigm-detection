"""Entry point of the `aigm` command.

Subcommands are registered here as they are implemented; nothing is listed before it works.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from aigm import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aigm",
        description="Experiments on detecting AI-generated music.",
    )
    parser.add_argument("--version", action="version", version=f"aigm {__version__}")
    parser.add_subparsers(dest="command", metavar="<command>")
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
