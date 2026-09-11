"""Smoke tests for the Step 1 foundation.

These verify the things every later step will rely on:
config loads, paths resolve to real places, logging is usable.
"""

from medimageforge import __version__
from medimageforge.config import PROJECT_ROOT, data_path, load_config


def test_version_exists():
    assert __version__


def test_default_config_loads():
    config = load_config()
    assert config["project"]["name"] == "medimageforge"
    assert "paths" in config and "dataset" in config


def test_configured_paths_resolve_inside_project():
    config = load_config()
    for key in config["paths"]:
        p = data_path(config, key)
        assert p.is_absolute()
        assert str(p).startswith(str(PROJECT_ROOT))


def test_dataset_is_present():
    # The CT-ICH dataset must be in place for every later step.
    config = load_config()
    assert data_path(config, "raw_dir").is_dir()
    assert data_path(config, "labels_csv").is_file()
