"""Command-line entry point: `python -m medimageforge <command>`.

Why this module exists:
    Every pipeline step becomes a subcommand here (`info` today; `explore`,
    `ingest`, `curate`, `qc`, ... in later steps). One entry point means one
    place where config is loaded and logging is configured — the "front door"
    of the platform.
"""

from __future__ import annotations

import argparse

from medimageforge import __version__
from medimageforge.config import data_path, load_config
from medimageforge.logging_utils import configure_logging, get_logger

log = get_logger(__name__)


def cmd_info(config: dict) -> int:
    """Print the resolved configuration and check which paths exist.

    This is our smoke test: if `info` works, config loading and path
    resolution work, and we can see whether the dataset is in place.
    """
    print(f"medimageforge {__version__}")
    print(f"log level: {config['logging']['level']}")
    print("\nConfigured paths:")
    for key in config["paths"]:
        p = data_path(config, key)
        status = "OK" if p.exists() else "MISSING"
        print(f"  [{status:7s}] {key}: {p}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="medimageforge")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file (default: configs/default.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="Show resolved configuration and path status")
    args = parser.parse_args()

    config = load_config(args.config)
    configure_logging(config["logging"]["level"])
    log.debug("Loaded config: %s", config)

    if args.command == "info":
        return cmd_info(config)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
