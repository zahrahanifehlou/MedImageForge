"""Tests for the Step 7 label store.

Emphasis on the properties the schema was designed for: multi-label support,
provenance, derived-not-stored values, and coexistence of several annotators
(which Step 12 will need when a model starts proposing labels).
"""

import pandas as pd
import pytest

from medimageforge.annotations import (
    DEFAULT_ANNOTATOR,
    HEMORRHAGE_TYPES,
    TAXONOMY,
    get_slice_annotation,
    label_distribution,
    load_annotations,
)
from medimageforge.config import data_path, load_config
from medimageforge.manifest import connect, ingest

WINDOWS = ["brain", "bone"]
MASK_SUFFIX = "_HGE_Seg"
COLUMNS = [
    "PatientNumber", "SliceNumber", "Intraventricular", "Intraparenchymal",
    "Subarachnoid", "Epidural", "Subdural", "No_Hemorrhage", "Fracture_Yes_No",
]


@pytest.fixture
def store(tmp_path):
    """A tiny manifest plus a 3-row label CSV covering the interesting cases."""
    data = tmp_path / "data"
    brain = data / "Patients_CT" / "049" / "brain"
    brain.mkdir(parents=True)
    for name in ("1.jpg", "14.jpg", f"14{MASK_SUFFIX}.jpg", "21.jpg", f"21{MASK_SUFFIX}.jpg"):
        (brain / name).write_bytes(name.encode())

    db = tmp_path / "manifest.db"
    ingest(data, db, WINDOWS, MASK_SUFFIX)

    labels = pd.DataFrame(
        [
            # normal slice: nothing positive
            [49, 1, 0, 0, 0, 0, 0, 1, 0],
            # epidural + fracture, has a mask
            [49, 14, 0, 0, 0, 1, 0, 0, 1],
            # TWO hemorrhage types at once, has a mask
            [49, 21, 1, 0, 0, 0, 1, 0, 0],
        ],
        columns=COLUMNS,
    )
    csv = tmp_path / "hemorrhage_diagnosis.csv"
    labels.to_csv(csv, index=False)
    return db, labels, csv


# ---------------------------------------------------------------------------
# taxonomy design
# ---------------------------------------------------------------------------

def test_no_hemorrhage_is_not_in_the_taxonomy():
    """It is derived (NOT any type), so storing it would allow disagreement."""
    codes = {t["code"] for t in TAXONOMY}
    assert "no_hemorrhage" not in codes
    assert not any(t["source_column"] == "No_Hemorrhage" for t in TAXONOMY)


def test_fracture_is_a_separate_category():
    """74 real slices have a fracture and no hemorrhage — not a bleed type."""
    fracture = next(t for t in TAXONOMY if t["code"] == "fracture")
    assert fracture["category"] == "other_finding"
    assert "fracture" not in HEMORRHAGE_TYPES
    assert len(HEMORRHAGE_TYPES) == 5


def test_loading_installs_the_taxonomy(store):
    db, labels, csv = store
    load_annotations(db, labels, csv, 3)
    with connect(db) as conn:
        rows = conn.execute("SELECT code, category FROM label_taxonomy").fetchall()
    assert len(rows) == len(TAXONOMY)


# ---------------------------------------------------------------------------
# translation from the wide CSV
# ---------------------------------------------------------------------------

def test_each_row_becomes_one_annotation_and_six_assertions(store):
    db, labels, csv = store
    report = load_annotations(db, labels, csv, 3)
    assert report.annotations == 3
    assert report.labels == 3 * len(TAXONOMY)
    assert report.positive_slices == 2
    assert report.with_mask == 2


def test_multi_label_slice_is_preserved(store):
    """The wide CSV allows several positives per row; so must the store."""
    db, labels, csv = store
    load_annotations(db, labels, csv, 3)
    record = get_slice_annotation(db, "049", 21)[0]
    assert sorted(record["hemorrhage_types"]) == ["intraventricular", "subdural"]
    assert record["no_hemorrhage"] is False


def test_labels_and_findings_are_distinguishable(store):
    db, labels, csv = store
    load_annotations(db, labels, csv, 3)
    record = get_slice_annotation(db, "049", 14)[0]
    assert record["hemorrhage_types"] == ["epidural"]
    assert "fracture" in record["positive_labels"]
    assert "fracture" not in record["hemorrhage_types"]


def test_no_hemorrhage_is_derived_on_read(store):
    db, labels, csv = store
    load_annotations(db, labels, csv, 3)
    assert get_slice_annotation(db, "049", 1)[0]["no_hemorrhage"] is True
    assert get_slice_annotation(db, "049", 14)[0]["no_hemorrhage"] is False


# ---------------------------------------------------------------------------
# the Done criterion: one call
# ---------------------------------------------------------------------------

def test_one_call_returns_labels_mask_and_provenance(store):
    db, labels, csv = store
    load_annotations(db, labels, csv, 3)
    record = get_slice_annotation(db, "049", 14)[0]

    assert record["labels"]                                  # labels
    assert record["mask_rel_path"].endswith("_HGE_Seg.jpg")   # mask reference
    prov = record["provenance"]                               # provenance
    assert prov["source_file"] == "hemorrhage_diagnosis.csv"
    assert len(prov["source_sha256"]) == 64
    assert prov["annotator"] == DEFAULT_ANNOTATOR
    assert prov["annotated_at"]


def test_slice_without_mask_has_none(store):
    db, labels, csv = store
    load_annotations(db, labels, csv, 3)
    assert get_slice_annotation(db, "049", 1)[0]["mask_rel_path"] is None


def test_unknown_slice_returns_empty(store):
    db, labels, csv = store
    load_annotations(db, labels, csv, 3)
    assert get_slice_annotation(db, "049", 999) == []


# ---------------------------------------------------------------------------
# provenance: why (patient, slice) is NOT the identity
# ---------------------------------------------------------------------------

def test_two_annotators_coexist_for_the_same_slice(store):
    """Step 12 will add model-proposed labels next to the radiologists'.

    Both must be stored and remain distinguishable — which is only possible
    because annotator is part of the annotation's identity.
    """
    db, labels, csv = store
    load_annotations(db, labels, csv, 3)
    # a "model" disagrees: it says subdural, not epidural
    model_view = labels.copy()
    model_view.loc[model_view.SliceNumber == 14, ["Epidural", "Subdural"]] = [0, 1]
    load_annotations(db, model_view, csv, 3, annotator="model-v0")

    records = get_slice_annotation(db, "049", 14)
    assert len(records) == 2
    by_annotator = {r["provenance"]["annotator"]: r["hemorrhage_types"] for r in records}
    assert by_annotator[DEFAULT_ANNOTATOR] == ["epidural"]
    assert by_annotator["model-v0"] == ["subdural"]

    # and we can ask for just one of them
    only = get_slice_annotation(db, "049", 14, annotator="model-v0")
    assert len(only) == 1


def test_reload_is_idempotent(store):
    db, labels, csv = store
    load_annotations(db, labels, csv, 3)
    load_annotations(db, labels, csv, 3)
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) c FROM slice_annotations").fetchone()["c"] == 3
        assert conn.execute("SELECT COUNT(*) c FROM slice_labels").fetchone()["c"] == 18


def test_corrected_label_updates_in_place(store):
    """Re-loading a corrected CSV must revise the value, not duplicate the row."""
    db, labels, csv = store
    load_annotations(db, labels, csv, 3)
    corrected = labels.copy()
    corrected.loc[corrected.SliceNumber == 1, "Subdural"] = 1
    load_annotations(db, corrected, csv, 3)

    record = get_slice_annotation(db, "049", 1)[0]
    assert record["hemorrhage_types"] == ["subdural"]
    with connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) c FROM slice_annotations").fetchone()["c"] == 3


# ---------------------------------------------------------------------------
# the real dataset
# ---------------------------------------------------------------------------

def test_real_store_matches_measured_counts():
    config = load_config()
    db = data_path(config, "manifest_db")
    if not db.is_file():
        pytest.skip("run ingest + load-labels first")
    with connect(db) as conn:
        tables = {
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "slice_annotations" not in tables:
            pytest.skip("run `python -m medimageforge load-labels` first")
        total = conn.execute("SELECT COUNT(*) c FROM slice_annotations").fetchone()["c"]
    if total == 0:
        pytest.skip("label store is empty")

    assert total == 2501
    # counts must equal what Step 2 measured straight from the CSV
    dist = {r["code"]: r["positives"] for r in label_distribution(db)}
    assert dist == {
        "epidural": 173,
        "intraparenchymal": 73,
        "intraventricular": 24,
        "subarachnoid": 18,
        "subdural": 56,
        "fracture": 195,
    }
