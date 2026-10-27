"""Command line entry point: ``neuromesh``."""

from __future__ import annotations

import argparse

from neuromesh import __version__
from neuromesh.runner.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="neuromesh")
    parser.add_argument("--version", action="version", version=f"neuromesh {__version__}")
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="run an experiment config")
    run.add_argument("config")
    run.add_argument("--dry-run", action="store_true", help="list runs without executing")
    args = parser.parse_args(argv)

    if args.command == "run":
        cfg = load_config(args.config)
        runs = cfg.matrix()
        print(f"{cfg.name}: {len(runs)} runs")
        if args.dry_run:
            return 0
        raise NotImplementedError("Track D: experiment execution")
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
