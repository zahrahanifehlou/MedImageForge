"""Tests for the Step 8 quality gates.

The real dataset produces 0 errors, so every gate is exercised here against
synthetic data that is deliberately broken — including a genuine
cross-patient duplicate, which the real data does not contain.
"""

import hashlib

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from medimageforge.config import data_path, load_config
from medimageforge.manifest import connect, ingest
from medimageforge.qc import (
    QCReport,
    check_counts,
    check_demographics,
    check_mask_label_agreement,
    check_orphans,
    correlation,
    dhash,
    find_leakage,
    hamming_matrix,
    run_qc,
)

WINDOWS = ["brain", "bone"]
MASK_SUFFIX = "_HGE_Seg"
LABEL_COLS = [
    "PatientNumber", "SliceNumber", "Intraventricular", "Intraparenchymal",
    "Subarachnoid", "Epidural", "Subdural", "No_Hemorrhage", "Fracture_Yes_No",
]
DEMO_COLS = [
    "Patient Number", "Age (years)", "Gender", "Intraventricular",
    "Intraparenchymal", "Subarachnoid", "Epidural", "Subdural",
]


def _img(seed, size=(64, 64)):
    """A SMOOTH, blob-structured image — deliberately not random noise.

    Perceptual hashing describes local gradient structure, so its stability
    depends on the image having structure. Measured under JPEG re-compression
    (quality 95 vs 40):

        pure random noise   dhash distance 4   (unstable)
        smooth blobs        dhash distance 1
        a real CT slice     dhash distance 0   (perfectly stable)

    An early version of these tests used `rng.integers(...)` noise and
    "failed" — not because the check was wrong, but because the fixture had
    nothing in common with a CT scan. A fixture must share the statistical
    properties of the real data or it tests the wrong thing.
    """
    rng = np.random.default_rng(seed)
    height, width = size
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    image = np.zeros((height, width), dtype=np.float32)
    for _ in range(5):
        centre_y, centre_x = rng.uniform(0, height), rng.uniform(0, width)
        spread = rng.uniform(height / 8, height / 3)
        amplitude = rng.uniform(60, 200)
        image += amplitude * np.exp(
            -(((grid_y - centre_y) ** 2 + (grid_x - centre_x) ** 2) / (2 * spread**2))
        )
    return np.clip(image, 0, 255).astype(np.uint8)


def _save(path, array, quality=95):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="JPEG", quality=quality)


# ---------------------------------------------------------------------------
# perceptual hashing — why it beats SHA-256 for this job
# ---------------------------------------------------------------------------

def test_dhash_is_stable_under_recompression(tmp_path):
    """THE reason we do not use SHA-256 for near-duplicate detection.

    The same image saved at two JPEG qualities has different bytes — so its
    cryptographic hash differs completely — but it looks identical, and dhash
    recognizes that.
    """
    array = _img(1, size=(256, 256))
    high, low = tmp_path / "q95.jpg", tmp_path / "q40.jpg"
    _save(high, array, quality=95)
    _save(low, array, quality=40)

    sha_high = hashlib.sha256(high.read_bytes()).hexdigest()
    sha_low = hashlib.sha256(low.read_bytes()).hexdigest()
    assert sha_high != sha_low                     # bytes differ

    distance = int((dhash(high) != dhash(low)).sum())
    assert distance <= 2                           # appearance does not


def test_dhash_differs_for_different_images(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    _save(a, _img(1, size=(256, 256)))
    _save(b, _img(2, size=(256, 256)))
    assert int((dhash(a) != dhash(b)).sum()) > 8


def test_dhash_length_is_64_bits(tmp_path):
    path = tmp_path / "a.jpg"
    _save(path, _img(1))
    assert dhash(path).shape == (64,)


def test_hamming_matrix_matches_bruteforce():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(12, 64)).astype(bool)
    matrix = hamming_matrix(bits)
    for i in range(12):
        for j in range(12):
            assert matrix[i, j] == int((bits[i] != bits[j]).sum())


def test_correlation_of_identical_vectors_is_one():
    vec = np.array([1.0, -1.0, 1.0, -1.0])
    assert correlation(vec, vec) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dataset(tmp_path):
    """Two patients, clean and consistent: the baseline every check passes."""
    data = tmp_path / "data"
    for patient, slices in (("049", [1, 2]), ("050", [1, 2])):
        for window in WINDOWS:
            for slice_no in slices:
                _save(data / "Patients_CT" / patient / window / f"{slice_no}.jpg",
                      _img(hash((patient, window, slice_no)) % 10_000))
    # patient 049 slice 2 is a hemorrhage: it needs a mask AND a positive label
    _save(data / "Patients_CT" / "049" / "brain" / f"2{MASK_SUFFIX}.jpg", _img(99))

    db = tmp_path / "manifest.db"
    ingest(data, db, WINDOWS, MASK_SUFFIX)

    labels = pd.DataFrame(
        [
            [49, 1, 0, 0, 0, 0, 0, 1, 0],
            [49, 2, 0, 0, 0, 1, 0, 0, 0],   # epidural + has mask
            [50, 1, 0, 0, 0, 0, 0, 1, 0],
            [50, 2, 0, 0, 0, 0, 0, 1, 0],
        ],
        columns=LABEL_COLS,
    )
    demographics = pd.DataFrame(
        [[49, 35, "Male", 0, 0, 0, 1, 0], [50, 40, "Female", 0, 0, 0, 0, 0]],
        columns=DEMO_COLS,
    )
    return data, db, labels, demographics


def test_clean_dataset_passes_every_gate(dataset):
    data, db, labels, demographics = dataset
    report, _ = run_qc(db, data, labels, demographics, 3, 2, 0.99)
    assert report.passed
    assert report.n_errors == 0


# ---------------------------------------------------------------------------
# each gate must fire
# ---------------------------------------------------------------------------

def test_window_count_mismatch_is_a_warning(dataset):
    """The real finding: patient 084 has 36 brain but 35 bone slices."""
    data, db, labels, demographics = dataset
    (data / "Patients_CT" / "049" / "bone" / "2.jpg").unlink()
    ingest(data, db, WINDOWS, MASK_SUFFIX)

    with connect(db) as conn:
        errors, warnings = check_counts(conn)
    assert errors == []                                   # usable data
    assert warnings[0]["issue"] == "window-count-mismatch"
    assert (warnings[0]["brain"], warnings[0]["bone"]) == (2, 1)


def test_label_without_image_is_an_error(dataset):
    data, db, labels, demographics = dataset
    extra = pd.DataFrame([[49, 99, 0, 0, 0, 0, 0, 1, 0]], columns=LABEL_COLS)
    with connect(db) as conn:
        errors, _ = check_orphans(conn, pd.concat([labels, extra]), 3)
    assert any(e["issue"] == "label-without-image" for e in errors)


def test_image_without_label_is_an_error(dataset):
    data, db, labels, demographics = dataset
    with connect(db) as conn:
        errors, _ = check_orphans(conn, labels[labels.SliceNumber != 1], 3)
    assert any(e["issue"] == "image-without-label" for e in errors)


def test_orphan_mask_is_an_error(dataset):
    data, db, labels, demographics = dataset
    _save(data / "Patients_CT" / "049" / "brain" / f"77{MASK_SUFFIX}.jpg", _img(7))
    ingest(data, db, WINDOWS, MASK_SUFFIX)
    with connect(db) as conn:
        errors, _ = check_orphans(conn, labels, 3)
    assert any(e["issue"] == "mask-without-slice" for e in errors)


def test_mask_contradicting_its_label_is_an_error(dataset):
    """A mask says 'hemorrhage here'; the label says there is none."""
    data, db, labels, demographics = dataset
    flipped = labels.copy()
    flipped.loc[(flipped.PatientNumber == 49) & (flipped.SliceNumber == 2), "Epidural"] = 0
    with connect(db) as conn:
        errors, _ = check_mask_label_agreement(conn, flipped, 3)
    assert any(e["issue"] == "mask-but-no-hemorrhage-label" for e in errors)


def test_hemorrhage_label_without_a_mask_is_an_error(dataset):
    data, db, labels, demographics = dataset
    (data / "Patients_CT" / "049" / "brain" / f"2{MASK_SUFFIX}.jpg").unlink()
    ingest(data, db, WINDOWS, MASK_SUFFIX)
    with connect(db) as conn:
        errors, _ = check_mask_label_agreement(conn, labels, 3)
    assert any(e["issue"] == "hemorrhage-label-but-no-mask" for e in errors)


# ---------------------------------------------------------------------------
# demographics
# ---------------------------------------------------------------------------

def test_missing_demographics_row_is_an_error(dataset):
    data, db, labels, demographics = dataset
    with connect(db) as conn:
        errors, _ = check_demographics(
            conn, demographics[demographics["Patient Number"] != 50], labels, 3
        )
    assert any(e["issue"] == "images-without-demographics" for e in errors)


def test_implausible_age_is_an_error(dataset):
    data, db, labels, demographics = dataset
    bad = demographics.copy()
    bad.loc[0, "Age (years)"] = 300
    with connect(db) as conn:
        errors, _ = check_demographics(conn, bad, labels, 3)
    assert any(e["issue"] == "implausible-age" for e in errors)


def test_neonate_age_is_a_warning_not_an_error(dataset):
    """Real finding: patient 085 is ~1 day old. Genuine data, so a warning.

    An IQR rule would not flag it at all — measured bounds on the real
    dataset are [-31.9, 83.1] years.
    """
    data, db, labels, demographics = dataset
    young = demographics.copy()
    young["Age (years)"] = young["Age (years)"].astype(float)
    young.loc[0, "Age (years)"] = 0.0027
    with connect(db) as conn:
        errors, warnings = check_demographics(conn, young, labels, 3)
    assert errors == []
    assert any(w["issue"] == "paediatric-age-under-1-year" for w in warnings)


def test_patient_vs_slice_label_disagreement_is_an_error(dataset):
    """Cross-validates the two CSVs against each other."""
    data, db, labels, demographics = dataset
    lying = demographics.copy()
    lying.loc[lying["Patient Number"] == 49, "Subdural"] = 1  # no slice says so
    with connect(db) as conn:
        errors, _ = check_demographics(conn, lying, labels, 3)
    assert any(e["issue"] == "patient-vs-slice-label-disagreement" for e in errors)


def test_blank_demographics_cells_count_as_zero(dataset):
    """The real CSV leaves negatives blank; NaN must not read as a positive."""
    data, db, labels, demographics = dataset
    sparse = demographics.copy()
    sparse[["Intraventricular", "Subarachnoid", "Subdural"]] = np.nan
    with connect(db) as conn:
        errors, _ = check_demographics(conn, sparse, labels, 3)
    assert not any(e["issue"] == "patient-vs-slice-label-disagreement" for e in errors)


# ---------------------------------------------------------------------------
# leakage — the check that needed calibration
# ---------------------------------------------------------------------------

def test_leakage_detects_the_same_scan_under_two_patients(dataset):
    """The failure that would invalidate every metric in Phase 4."""
    data, db, labels, demographics = dataset
    shared = _img(4242, size=(256, 256))
    _save(data / "Patients_CT" / "049" / "brain" / "3.jpg", shared, quality=95)
    # same image, re-compressed: different bytes, identical appearance
    _save(data / "Patients_CT" / "050" / "brain" / "3.jpg", shared, quality=60)
    ingest(data, db, WINDOWS, MASK_SUFFIX)

    with connect(db) as conn:
        errors, _, stats = find_leakage(conn, data, max_hamming=2, min_correlation=0.99)
    assert any(e["issue"] == "cross-patient-near-duplicate" for e in errors)
    assert stats["max_correlation"] >= 0.99


def test_leakage_ignores_merely_similar_slices(dataset):
    """Anatomical similarity is not duplication.

    On the real dataset the closest cross-patient pair scores 0.950 while
    same-patient adjacent slices average 0.896 — overlapping distributions.
    The gate must not treat "looks alike" as "is the same".
    """
    data, db, labels, demographics = dataset
    # Calibrated to land BETWEEN the two stages: mild additive noise keeps the
    # gradient structure (dhash distance 2, so stage 1 proposes the pair) while
    # correlation is 0.99996 — high, but short of an exact-duplicate threshold.
    base = _img(7, size=(256, 256)).astype(np.float32)
    rng = np.random.default_rng(0)
    variant = np.clip(base + rng.normal(0, 2, base.shape), 0, 255).astype(np.uint8)
    _save(data / "Patients_CT" / "049" / "brain" / "4.jpg", base.astype(np.uint8))
    _save(data / "Patients_CT" / "050" / "brain" / "4.jpg", variant)
    ingest(data, db, WINDOWS, MASK_SUFFIX)

    with connect(db) as conn:
        errors, _, stats = find_leakage(conn, data, max_hamming=2, min_correlation=0.999999)
    assert stats["candidates"] >= 1      # stage 1 proposes it
    assert errors == []                  # stage 2 rejects it


def test_leakage_can_be_skipped(dataset):
    data, db, labels, demographics = dataset
    report, stats = run_qc(db, data, labels, demographics, 3, 2, 0.99, skip_leakage=True)
    assert stats == {}
    assert any(c["check"] == "leakage" and "skipped" in c["detail"] for c in report.checks)


# ---------------------------------------------------------------------------
# gate semantics
# ---------------------------------------------------------------------------

def test_warnings_do_not_fail_the_gate():
    report = QCReport()
    report.add("x", "d", errors=[], warnings=[{"issue": "noted"}])
    assert report.passed
    assert report.n_warnings == 1


def test_errors_fail_the_gate():
    report = QCReport()
    report.add("x", "d", errors=[{"issue": "bad"}], warnings=[])
    assert not report.passed
    assert report.as_dict()["gate"] == "FAIL"


# ---------------------------------------------------------------------------
# the real dataset
# ---------------------------------------------------------------------------

@pytest.mark.needs_data
def test_real_dataset_has_no_errors_but_finds_known_warnings():
    config = load_config()
    db = data_path(config, "manifest_db")
    if not db.is_file():
        pytest.skip("run `python -m medimageforge ingest` first")

    from medimageforge.explore import load_demographics, load_labels

    report, _ = run_qc(
        db,
        data_path(config, "data_dir"),
        load_labels(data_path(config, "labels_csv")),
        load_demographics(data_path(config, "demographics_csv")),
        config["dataset"]["patient_id_width"],
        config["qc"]["leakage_max_hamming"],
        config["qc"]["leakage_min_correlation"],
        skip_leakage=True,
    )
    assert report.passed
    issues = {w["issue"] for c in report.checks for w in c["warnings"]}
    # the two real inconsistencies this dataset actually contains
    assert "window-count-mismatch" in issues
    assert "paediatric-age-under-1-year" in issues
