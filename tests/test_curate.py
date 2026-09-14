"""Tests for the Step 5 curation pipeline.

The real dataset is clean, so it never exercises a rejection. These tests
build deliberately broken files to prove each gate actually fires — an
untested gate is an assumption, not a control.
"""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from medimageforge.config import data_path, load_config
from medimageforge.curate import (
    curate,
    curated_rel_path,
    curation_state,
    normalize_and_save,
    validate_image,
)
from medimageforge.imaging import load_gray
from medimageforge.manifest import connect, ingest

WINDOWS = ["brain", "bone"]
MASK_SUFFIX = "_HGE_Seg"
SIZE = (650, 650)


def _write_jpg(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="JPEG", quality=95)


def _noise(size=SIZE, seed=0):
    """A non-blank image — passes the 'has information content' rule."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=size, dtype=np.uint8)


# --------------------------------------------------------------------------
# validate_image: one test per rule
# --------------------------------------------------------------------------

def test_validate_accepts_a_good_image(tmp_path):
    path = tmp_path / "good.jpg"
    _write_jpg(path, _noise())
    reasons, array = validate_image(path, SIZE)
    assert reasons == []
    assert array.shape == SIZE


def test_validate_rejects_wrong_dimensions(tmp_path):
    path = tmp_path / "small.jpg"
    _write_jpg(path, _noise(size=(64, 64)))
    reasons, _ = validate_image(path, SIZE)
    assert any(r.startswith("wrong-dimensions") for r in reasons)


def test_validate_rejects_blank_image(tmp_path):
    path = tmp_path / "black.jpg"
    _write_jpg(path, np.zeros(SIZE, dtype=np.uint8))
    reasons, _ = validate_image(path, SIZE)
    assert "blank-or-constant" in reasons


def test_validate_rejects_unreadable_file(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"this is not an image")
    reasons, array = validate_image(path, SIZE)
    assert array is None
    assert any(r.startswith("unreadable") for r in reasons)


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------

def test_mask_normalization_produces_true_binary(tmp_path):
    """The defect we found in Step 3, fixed once and centrally."""
    noisy = np.array([[0, 7, 200], [3, 255, 90]], dtype=np.uint8)
    out = tmp_path / "m.png"
    normalize_and_save(noisy, out, is_mask=True, mask_threshold=128)
    assert sorted(np.unique(load_gray(out)).tolist()) == [0, 255]


def test_slice_normalization_is_lossless(tmp_path):
    array = _noise(size=(32, 32))
    out = tmp_path / "s.png"
    normalize_and_save(array, out, is_mask=False, mask_threshold=128)
    assert np.array_equal(load_gray(out), array)  # PNG round-trips exactly


def test_curated_paths_have_no_jpeg():
    assert curated_rel_path("049", "brain", 14, False) == "049/brain/14.png"
    assert curated_rel_path("049", "brain", 14, True) == "049/brain/14_mask.png"


# --------------------------------------------------------------------------
# the pipeline end to end, on a synthetic dataset
# --------------------------------------------------------------------------

@pytest.fixture
def broken_dataset(tmp_path):
    """A raw zone containing one of every failure mode."""
    data = tmp_path / "data"
    brain = data / "Patients_CT" / "049" / "brain"
    bone = data / "Patients_CT" / "049" / "bone"

    _write_jpg(brain / "1.jpg", _noise(seed=1))          # good, paired
    _write_jpg(bone / "1.jpg", _noise(seed=2))           # good, paired
    _write_jpg(brain / "2.jpg", _noise(seed=3))          # good, NO bone pair -> warning
    _write_jpg(brain / "3.jpg", _noise(size=(64, 64)))   # wrong dimensions -> reject
    _write_jpg(brain / "4.jpg", np.zeros(SIZE, np.uint8))  # blank -> reject
    (brain / "5.jpg").write_bytes(b"garbage")            # unreadable -> reject
    _write_jpg(brain / "6.jpg", _noise(seed=1))          # byte-identical to 1.jpg -> duplicate
    _write_jpg(brain / f"1{MASK_SUFFIX}.jpg", _noise(seed=4))  # mask for slice 1: ok
    _write_jpg(brain / f"9{MASK_SUFFIX}.jpg", _noise(seed=5))  # mask, no slice 9 -> reject

    labels = pd.DataFrame(
        # slice 7 has no file; slice 6 deliberately has no label row
        {"PatientNumber": [49] * 6, "SliceNumber": [1, 2, 3, 4, 5, 7]}
    )
    db = tmp_path / "artifacts" / "manifest.db"
    ingest(data, db, WINDOWS, MASK_SUFFIX)
    return data, tmp_path / "curated", db, labels


def _decisions(db):
    with connect(db) as conn:
        return {
            r["rel_path"]: (r["decision"], r["reasons"])
            for r in conn.execute("SELECT rel_path, decision, reasons FROM curation")
        }


def test_every_rejection_gate_fires(broken_dataset):
    data, curated, db, labels = broken_dataset
    report = curate(data, curated, db, labels, SIZE, 128)
    d = _decisions(db)

    assert d["Patients_CT/049/brain/3.jpg"][0] == "rejected"
    assert "wrong-dimensions" in d["Patients_CT/049/brain/3.jpg"][1]
    assert "blank-or-constant" in d["Patients_CT/049/brain/4.jpg"][1]
    assert "unreadable" in d["Patients_CT/049/brain/5.jpg"][1]
    # 6.jpg is byte-identical to 1.jpg AND has no label row
    assert "duplicate-of" in d["Patients_CT/049/brain/6.jpg"][1]
    assert "mask-without-slice" in d[f"Patients_CT/049/brain/9{MASK_SUFFIX}.jpg"][1]
    assert report.rejected == 5
    # every rejection is reported with a reason — nothing dropped silently
    assert all(item["reasons"] for item in report.rejections)


def test_unpaired_slice_is_warned_not_rejected(broken_dataset):
    """Not every anomaly is a defect: a usable slice must survive curation."""
    data, curated, db, labels = broken_dataset
    curate(data, curated, db, labels, SIZE, 128)

    assert _decisions(db)["Patients_CT/049/brain/2.jpg"][0] == "accepted"
    assert (curated / "049" / "brain" / "2.png").is_file()


def test_curated_zone_contains_only_accepted_files(broken_dataset):
    data, curated, db, labels = broken_dataset
    curate(data, curated, db, labels, SIZE, 128)

    produced = {p.name for p in curated.rglob("*.png")}
    assert produced == {"1.png", "2.png", "1_mask.png"}
    assert not list(curated.rglob("*.jpg"))  # no lossy files in the curated zone


def test_curation_is_idempotent(broken_dataset):
    data, curated, db, labels = broken_dataset
    curate(data, curated, db, labels, SIZE, 128)
    second = curate(data, curated, db, labels, SIZE, 128)

    assert second.accepted == 0
    assert second.skipped > 0


def test_state_report_survives_an_idempotent_rerun(broken_dataset):
    """A report of one run forgets; a report of the state does not.

    After a no-op re-run, "this run" has zero findings — but the dataset
    still has the same rejections and warnings, and the report must say so.
    """
    data, curated, db, labels = broken_dataset
    curate(data, curated, db, labels, SIZE, 128)
    second = curate(data, curated, db, labels, SIZE, 128)
    assert second.rejections == [] and second.warnings == []  # run view: nothing

    state = curation_state(db)  # state view: still the truth
    assert state["rejected"] == 5
    # brain/1, bone/1, brain/2, brain/1_mask
    assert state["accepted"] == 4
    assert any(
        "missing-bone-counterpart" in w["warnings"] for w in state["warnings"]
    )


def test_raw_zone_is_never_modified(broken_dataset):
    """Zone separation: curation must not touch a single raw byte."""
    data, curated, db, labels = broken_dataset
    before = {p: p.read_bytes() for p in sorted(data.rglob("*")) if p.is_file()}
    curate(data, curated, db, labels, SIZE, 128)
    after = {p: p.read_bytes() for p in sorted(data.rglob("*")) if p.is_file()}
    assert before == after


def test_real_curation_matches_manifest():
    """On the real dataset: every image curated, only metadata left as raw."""
    config = load_config()
    db = data_path(config, "manifest_db")
    if not db.is_file():
        pytest.skip("run `python -m medimageforge ingest && ... curate` first")
    with connect(db) as conn:
        status = {
            r["status"]: r["n"]
            for r in conn.execute("SELECT status, COUNT(*) n FROM files GROUP BY status")
        }
    if "curated" not in status:
        pytest.skip("run `python -m medimageforge curate` first")
    assert status["curated"] == 5319
    assert status.get("raw") == 7  # the 7 metadata files are not images
