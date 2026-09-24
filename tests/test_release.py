"""Tests for the Step 9 dataset release.

The properties under test are the ones a release exists to guarantee:
patient-level splitting (no leakage), reproducibility, immutability,
pseudonymity, and detectability of drift.
"""

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from medimageforge.config import data_path, load_config
from medimageforge.manifest import connect, ingest
from medimageforge.privacy import pseudonymize
from medimageforge.release import (
    SPLITS,
    _stable_hash,
    assert_no_patient_overlap,
    build_index,
    patient_strata,
    render_dataset_card,
    split_patients,
    split_statistics,
    verify_release,
    write_release,
)

RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SALT = b"test-salt"
LABEL_COLS = [
    "PatientNumber", "SliceNumber", "Intraventricular", "Intraparenchymal",
    "Subarachnoid", "Epidural", "Subdural", "No_Hemorrhage", "Fracture_Yes_No",
]


def _labels(n_patients=40, slices=6, positive_every=2):
    """Synthetic labels: every `positive_every`-th patient has a hemorrhage."""
    rows = []
    for patient in range(1, n_patients + 1):
        positive = patient % positive_every == 0
        for slice_no in range(1, slices + 1):
            # only the first two slices of a positive patient are positive
            is_pos = positive and slice_no <= 2
            rows.append(
                [patient, slice_no, 0, int(is_pos), 0, 0, 0, int(not is_pos), 0]
            )
    return pd.DataFrame(rows, columns=LABEL_COLS)


# ---------------------------------------------------------------------------
# reproducibility — the bug this step is about
# ---------------------------------------------------------------------------

def test_stable_hash_is_pinned_to_known_values():
    """Guards against re-introducing Python's per-process randomized hash().

    These constants were computed once. If someone swaps _stable_hash for
    hash(), this test fails immediately instead of silently changing every
    future split.
    """
    assert _stable_hash("True") == 1018988487
    assert _stable_hash("False") == 1621311084
    assert _stable_hash("hemorrhage") == 3935587066


def test_same_seed_gives_the_same_split():
    strata = patient_strata(_labels(), 3)
    first = split_patients(strata, RATIOS, 20260915)
    second = split_patients(strata, RATIOS, 20260915)
    assert first == second


def test_different_seed_gives_a_different_split():
    strata = patient_strata(_labels(), 3)
    assert split_patients(strata, RATIOS, 1) != split_patients(strata, RATIOS, 2)


def test_split_is_independent_of_input_ordering():
    """Reproducible from the manifest means insensitive to row order."""
    labels = _labels()
    strata = patient_strata(labels, 3)
    shuffled = dict(reversed(list(strata.items())))
    assert split_patients(strata, RATIOS, 7) == split_patients(shuffled, RATIOS, 7)


# ---------------------------------------------------------------------------
# the anti-leakage rule
# ---------------------------------------------------------------------------

def test_every_patient_lands_in_exactly_one_split():
    strata = patient_strata(_labels(), 3)
    assignment = split_patients(strata, RATIOS, 20260915)
    assert set(assignment) == set(strata)
    assert set(assignment.values()) <= set(SPLITS)
    assert_no_patient_overlap(assignment)


def test_overlap_detector_actually_raises():
    """A guarantee that is never tested is not a guarantee."""

    class Overlapping(dict):
        def items(self):  # pragma: no cover - simple stub
            return [("049", "train"), ("049", "test")]

    with pytest.raises(ValueError, match="both"):
        assert_no_patient_overlap(Overlapping())


def test_split_is_by_patient_not_by_slice(tmp_path):
    """No patient's slices may appear in two splits."""
    data, db, labels, pseudonyms = _dataset(tmp_path)
    strata = patient_strata(labels, 3)
    assignment = split_patients(strata, RATIOS, 20260915)
    index = build_index(db, assignment, pseudonyms, labels, 3)

    per_patient = index.groupby("patient")["split"].nunique()
    assert (per_patient == 1).all()


# ---------------------------------------------------------------------------
# stratification and its limits
# ---------------------------------------------------------------------------

def test_stratification_spreads_positive_patients_across_splits():
    strata = patient_strata(_labels(n_patients=40, positive_every=2), 3)
    assignment = split_patients(strata, RATIOS, 20260915)
    for split in SPLITS:
        patients = [p for p, s in assignment.items() if s == split]
        positives = sum(1 for p in patients if strata[p]["has_hemorrhage"])
        assert positives > 0, f"{split} has no hemorrhage-positive patient"


def test_singleton_stratum_goes_to_train():
    """The real constraint: hemorrhage x fracture leaves a stratum of ONE.

    round(1 * 0.70) == 1, so the lone patient goes to train and the other
    splits get nothing from that stratum. Documented, not accidental.
    """
    strata = {
        "001": {"has_hemorrhage": True, "has_fracture": True, "slices": 5, "positive_slices": 2},
    }
    assignment = split_patients(strata, RATIOS, 1)
    assert assignment == {"001": "train"}


def test_patient_stratification_does_not_guarantee_slice_balance(tmp_path):
    """Why the card reports slice rates instead of promising them.

    A positive patient contributes between 1 and 19 positive slices in the
    real data, so balancing patients cannot balance slices.
    """
    data, db, labels, pseudonyms = _dataset(tmp_path)
    strata = patient_strata(labels, 3)
    assignment = split_patients(strata, RATIOS, 20260915)
    index = build_index(db, assignment, pseudonyms, labels, 3)
    stats = split_statistics(index, strata, assignment)
    rates = [stats[s]["hemorrhage_rate"] for s in SPLITS if stats[s]["brain_slices"]]
    assert all(0.0 <= r <= 1.0 for r in rates)   # reported, not asserted equal


# ---------------------------------------------------------------------------
# a real (tiny) release
# ---------------------------------------------------------------------------

def _dataset(tmp_path, n_patients=12):
    """A curated manifest for n patients, 4 brain + 4 bone slices each."""
    data = tmp_path / "data"
    curated = tmp_path / "curated"
    rng = np.random.default_rng(0)

    for patient in range(1, n_patients + 1):
        patient_id = f"{patient:03d}"
        for window in ("brain", "bone"):
            for slice_no in range(1, 5):
                array = rng.integers(0, 255, (16, 16), dtype=np.uint8)
                raw = data / "Patients_CT" / patient_id / window / f"{slice_no}.jpg"
                raw.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(array).save(raw, format="JPEG")
                out = curated / patient_id / window / f"{slice_no}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(array).save(out, format="PNG")

    db = tmp_path / "manifest.db"
    ingest(data, db, ["brain", "bone"], "_HGE_Seg")

    # Minimal curation records, as Step 5 would have written them.
    import hashlib

    from medimageforge.curate import CURATION_SCHEMA

    with connect(db) as conn:
        conn.executescript(CURATION_SCHEMA)
        for row in conn.execute(
            "SELECT rel_path, patient_id, window, slice_no FROM files WHERE kind='slice'"
        ).fetchall():
            out_rel = f"{row['patient_id']}/{row['window']}/{row['slice_no']}.png"
            digest = hashlib.sha256((curated / out_rel).read_bytes()).hexdigest()
            conn.execute(
                """INSERT INTO curation (rel_path, decision, reasons, warnings,
                       curated_path, source_sha256, curated_sha256, curated_at)
                   VALUES (?, 'accepted', '', '', ?, 'x', ?, 'now')""",
                (row["rel_path"], out_rel, digest),
            )
        conn.commit()

    labels = _labels(n_patients=n_patients, slices=4, positive_every=2)
    pseudonyms = {f"{p:03d}": pseudonymize(f"{p:03d}", SALT) for p in range(1, n_patients + 1)}
    return data, db, labels, pseudonyms


@pytest.fixture
def release(tmp_path):
    data, db, labels, pseudonyms = _dataset(tmp_path)
    strata = patient_strata(labels, 3)
    assignment = split_patients(strata, RATIOS, 20260915)
    index = build_index(db, assignment, pseudonyms, labels, 3)
    stats = split_statistics(index, strata, assignment)
    metadata = {
        "version": "v1.0",
        "created_at": "2026-01-01T00:00:00+00:00",
        "code_version": "0.1.0",
        "seed": 20260915,
        "ratios": RATIOS,
        "stratify_by": "hemorrhage",
        "labels_file": "hemorrhage_diagnosis.csv",
        "labels_sha256": "deadbeef",
    }
    out = tmp_path / "datasets" / "v1.0"
    write_release(out, index, assignment, pseudonyms, stats, metadata, ["an issue"])
    return out, tmp_path / "curated", pseudonyms, index


def test_release_contains_the_expected_files(release):
    out, _, _, _ = release
    names = {p.name for p in out.iterdir()}
    assert names == {
        "index.csv", "splits.csv", "metadata.json", "dataset_card.md", "CHECKSUMS.txt"
    }


def test_release_files_are_read_only(release):
    """Immutability, enforced on disk and not only by convention."""
    out, _, _, _ = release
    for path in out.iterdir():
        assert oct(path.stat().st_mode)[-3:] == "444"


def test_index_uses_pseudonyms_and_leaks_no_real_id(release):
    """A release leaves the controlled zone (Step 6), so no real IDs."""
    out, _, pseudonyms, _ = release
    text = (out / "index.csv").read_text()
    for real_id in pseudonyms:
        assert f",{real_id}/" not in text          # not in a path
        assert f",{real_id}," not in text          # not as a field
    index = pd.read_csv(out / "index.csv")
    assert index["patient"].str.startswith("PAT-").all()
    assert index["path"].str.startswith("PAT-").all()


def test_verify_passes_on_a_fresh_release(release):
    out, curated, pseudonyms, _ = release
    findings = verify_release(out, curated, pseudonyms)
    assert all(v == [] for v in findings.values())


def test_verify_detects_a_modified_release_file(release):
    out, curated, pseudonyms, _ = release
    path = out / "splits.csv"
    path.chmod(0o644)
    path.write_text("patient,split\nPAT-tampered,train\n")
    findings = verify_release(out, curated, pseudonyms)
    assert any(f["issue"] == "modified" for f in findings["release_files"])


def test_verify_detects_drift_in_a_referenced_image(release):
    """The point of storing a hash per file instead of copying the file."""
    out, curated, pseudonyms, index = release
    first = index.iloc[0]
    real_id = {v: k for k, v in pseudonyms.items()}[first["patient"]]
    target = curated / real_id / first["window"] / f"{first['slice_no']}.png"
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(target)

    findings = verify_release(out, curated, pseudonyms)
    assert any(f["issue"] == "content-changed" for f in findings["referenced_images"])


def test_verify_reports_missing_checksums(tmp_path):
    findings = verify_release(tmp_path, tmp_path, {})
    assert findings["release_files"][0]["issue"] == "missing-CHECKSUMS.txt"


def test_metadata_records_provenance_and_seed(release):
    out, _, _, _ = release
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["seed"] == 20260915
    assert metadata["labels_sha256"] == "deadbeef"
    assert metadata["stratify_by"] == "hemorrhage"
    assert "statistics" in metadata


def test_dataset_card_documents_counts_and_issues(release):
    out, _, _, _ = release
    card = (out / "dataset_card.md").read_text()
    assert "# Dataset card — v1.0" in card
    assert "Known issues" in card and "an issue" in card
    assert "SPLIT BY PATIENT" in card or "Unit: the patient" in card
    assert "pseudonym" in card.lower()


def test_card_renders_every_split():
    stats = {
        s: {
            "patients": 1, "patients_with_hemorrhage": 1, "patients_with_fracture": 0,
            "images": 2, "brain_slices": 1, "hemorrhage_slices": 1, "hemorrhage_rate": 1.0,
        }
        for s in SPLITS
    }
    metadata = {
        "created_at": "x", "code_version": "0.1.0", "labels_file": "f.csv",
        "labels_sha256": "abc", "ratios": RATIOS, "seed": 1,
    }
    card = render_dataset_card("v9.9", stats, metadata, ["issue one"])
    for split in SPLITS:
        assert split in card


# ---------------------------------------------------------------------------
# the real published release
# ---------------------------------------------------------------------------

@pytest.mark.needs_data
def test_real_release_has_no_patient_overlap_and_verifies():
    config = load_config()
    release_dir = data_path(config, "datasets_dir") / config["release"]["version"]
    if not (release_dir / "index.csv").is_file():
        pytest.skip("run `python -m medimageforge release` first")

    index = pd.read_csv(release_dir / "index.csv")
    assert (index.groupby("patient")["split"].nunique() == 1).all()
    assert index["patient"].nunique() == 82
    assert set(index["split"]) == set(SPLITS)
    # brain + bone slices of the curated zone
    assert len(index) == 5001
