"""Tests for the Step 2 dataset explorer.

The scanner and loader run against the real data/ (it is the fixture —
measuring it is the point). Checksum logic gets a tiny synthetic case so we
can test the failure modes the real dataset (hopefully) never triggers.
"""

import hashlib

import pytest

from medimageforge.config import data_path, load_config
from medimageforge.explore import (
    load_demographics,
    load_labels,
    scan_raw_dir,
    verify_checksums,
)


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_scan_finds_all_patients(config):
    patients = scan_raw_dir(
        data_path(config, "raw_dir"),
        config["dataset"]["windows"],
        config["dataset"]["mask_suffix"],
    )
    assert len(patients) == 82
    assert all(p.slice_counts["brain"] > 0 for p in patients)


def test_masks_are_not_counted_as_slices(config):
    patients = scan_raw_dir(
        data_path(config, "raw_dir"),
        config["dataset"]["windows"],
        config["dataset"]["mask_suffix"],
    )
    total_masks = sum(p.mask_count for p in patients)
    brain_slices = sum(p.slice_counts["brain"] for p in patients)
    # 318 real masks; if masks leaked into the slice count the total would differ.
    assert total_masks == 318
    assert brain_slices == 2501


def test_labels_parse_and_cover_every_brain_slice(config):
    labels = load_labels(data_path(config, "labels_csv"))
    assert "PatientNumber" in labels.columns  # BOM stripped correctly
    assert len(labels) == 2501
    assert labels["PatientNumber"].nunique() == 82


def test_demographics_parse(config):
    demographics = load_demographics(data_path(config, "demographics_csv"))
    assert len(demographics) == 82


def test_checksums_pass_on_real_data(config):
    report = verify_checksums(
        data_path(config, "data_dir"), data_path(config, "checksums_file")
    )
    assert report.mismatched == []
    assert report.missing == []
    # SHA256SUMS.txt cannot list its own hash — the only expected "extra".
    assert report.extra == ["SHA256SUMS.txt"]


def test_verify_checksums_detects_all_failure_modes(tmp_path):
    (tmp_path / "ok.jpg").write_bytes(b"good bytes")
    (tmp_path / "corrupt.jpg").write_bytes(b"tampered")
    ok_hash = hashlib.sha256(b"good bytes").hexdigest()
    sums = tmp_path / "SHA256SUMS.txt"
    sums.write_text(
        f"{ok_hash} ok.jpg\n"
        f"{'0' * 64} corrupt.jpg\n"
        f"{'0' * 64} gone.jpg\n"
    )
    (tmp_path / "stray.jpg").write_bytes(b"unlisted")

    report = verify_checksums(tmp_path, sums)
    assert report.verified == 1
    assert report.mismatched == ["corrupt.jpg"]
    assert report.missing == ["gone.jpg"]
    # The sums file never lists itself, so it is always reported "extra".
    assert sorted(report.extra) == ["SHA256SUMS.txt", "stray.jpg"]
    assert not report.ok
