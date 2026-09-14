"""Tests for the Step 4 ingestion pipeline and manifest.

Path classification and the ingest lifecycle run on a tiny synthetic data
tree — that lets us test the cases the real dataset never shows (a file
changing, a file disappearing) without touching raw data.
"""

from pathlib import Path

import pytest

from medimageforge.config import data_path, load_config
from medimageforge.manifest import classify, connect, ingest

WINDOWS = ["brain", "bone"]
MASK_SUFFIX = "_HGE_Seg"


def test_classify_slice_mask_and_metadata():
    slice_rec = classify(Path("Patients_CT/049/brain/14.jpg"), WINDOWS, MASK_SUFFIX)
    assert slice_rec == {
        "kind": "slice", "patient_id": "049", "window": "brain", "slice_no": 14
    }

    mask_rec = classify(
        Path("Patients_CT/049/brain/14_HGE_Seg.jpg"), WINDOWS, MASK_SUFFIX
    )
    assert mask_rec["kind"] == "mask"
    assert mask_rec["slice_no"] == 14  # mask points at the slice it belongs to

    meta = classify(Path("hemorrhage_diagnosis.csv"), WINDOWS, MASK_SUFFIX)
    assert meta["kind"] == "metadata"
    assert meta["patient_id"] is None


@pytest.fixture
def tiny_dataset(tmp_path):
    """A 3-file stand-in for data/: one slice, its mask, one metadata file."""
    brain = tmp_path / "data" / "Patients_CT" / "049" / "brain"
    brain.mkdir(parents=True)
    (brain / "14.jpg").write_bytes(b"slice bytes")
    (brain / "14_HGE_Seg.jpg").write_bytes(b"mask bytes")
    (tmp_path / "data" / "labels.csv").write_text("a,b\n1,2\n")
    return tmp_path / "data", tmp_path / "artifacts" / "manifest.db"


def test_first_ingest_registers_everything(tiny_dataset):
    data_dir, db = tiny_dataset
    report = ingest(data_dir, db, WINDOWS, MASK_SUFFIX)
    assert (report.total_on_disk, report.new, report.unchanged) == (3, 3, 0)

    with connect(db) as conn:
        kinds = {r["kind"] for r in conn.execute("SELECT kind FROM files")}
    assert kinds == {"slice", "mask", "metadata"}


def test_reingest_is_idempotent(tiny_dataset):
    """The core contract: re-running on unchanged data changes nothing."""
    data_dir, db = tiny_dataset
    ingest(data_dir, db, WINDOWS, MASK_SUFFIX)
    second = ingest(data_dir, db, WINDOWS, MASK_SUFFIX)

    assert second.new == 0
    assert second.unchanged == 3
    assert second.updated == 0
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 3


def test_changed_content_is_detected(tiny_dataset):
    data_dir, db = tiny_dataset
    ingest(data_dir, db, WINDOWS, MASK_SUFFIX)
    (data_dir / "Patients_CT" / "049" / "brain" / "14.jpg").write_bytes(b"TAMPERED!!")

    report = ingest(data_dir, db, WINDOWS, MASK_SUFFIX)
    assert report.updated == 1
    assert report.unchanged == 2


def test_vanished_file_is_marked_missing_not_deleted(tiny_dataset):
    """A manifest records history — it never silently forgets a file."""
    data_dir, db = tiny_dataset
    ingest(data_dir, db, WINDOWS, MASK_SUFFIX)
    (data_dir / "Patients_CT" / "049" / "brain" / "14_HGE_Seg.jpg").unlink()

    report = ingest(data_dir, db, WINDOWS, MASK_SUFFIX)
    assert report.missing == 1
    with connect(db) as conn:
        row = conn.execute(
            "SELECT status FROM files WHERE rel_path LIKE '%_HGE_Seg.jpg'"
        ).fetchone()
    assert row["status"] == "missing"  # row still there, flagged


def test_manifest_matches_explorer_counts():
    """The roadmap's Done criterion: manifest counts == Step 2 counts."""
    config = load_config()
    db = data_path(config, "manifest_db")
    if not db.is_file():
        pytest.skip("run `python -m medimageforge ingest` first")

    with connect(db) as conn:
        counts = {
            (r["kind"], r["window"]): r["n"]
            for r in conn.execute(
                "SELECT kind, window, COUNT(*) n FROM files GROUP BY kind, window"
            )
        }
        patients = conn.execute(
            "SELECT COUNT(DISTINCT patient_id) n FROM files WHERE patient_id IS NOT NULL"
        ).fetchone()["n"]

    assert patients == 82
    assert counts[("slice", "brain")] == 2501
    assert counts[("slice", "bone")] == 2500
    assert counts[("mask", "brain")] == 318
