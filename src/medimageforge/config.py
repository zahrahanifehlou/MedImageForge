"""Configuration loading and path resolution.

Why this module exists:
    Nothing in the codebase should hardcode a path like "data/Patients_CT".
    All settings live in configs/*.yaml so the same code runs on a laptop,
    in CI, or on a server by pointing at a different config file.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# src/medimageforge/config.py -> parents[2] is the repository root.
# This works because we install the package with `pip install -e .`, which
# keeps __file__ pointing at the real source tree.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Load a YAML config file into a plain dict.

    Args:
        path: Config file to load. Defaults to configs/default.yaml.

    Returns:
        The parsed config as nested dicts.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_path(config: dict, key: str) -> Path:
    """Resolve a paths.<key> entry to an absolute path under the project root."""
    return PROJECT_ROOT / config["paths"][key]
