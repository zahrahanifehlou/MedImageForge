"""The label store: annotations as first-class, queryable data.

Why this module exists:
    Labels are data too. In `hemorrhage_diagnosis.csv` they are seven boolean
    columns in a *wide* table — a shape that cannot express:

        - who made this annotation (a radiologist? a model? which one?)
        - when, and from which version of which file
        - a new label type, without ALTER TABLE

    So we translate the CSV into a small, normalized schema. The translation
    itself is the lesson: schema-before-storage.

Three tables, and why it is not one:

    label_taxonomy    the controlled vocabulary. Adding a label is an INSERT,
                      not a schema migration.
    slice_annotations one row per annotated slice: the mask reference plus
                      provenance. The mask depends on the SLICE, not on each
                      individual label — keeping it here avoids repeating it
                      once per label (a transitive dependency).
    slice_labels      one row per (annotation, label) assertion: the actual
                      multi-label payload.

A deliberate omission:
    The CSV's `No_Hemorrhage` column gets no taxonomy row. We measured it: it
    equals NOT(any hemorrhage type) in all 2501 rows. It is *derived*, so we
    compute it on read. Storing a value that is computable from others is an
    invitation for the two to disagree.

Why provenance is not optional:
    In Step 12 a model will propose labels for the same slices these
    radiologists annotated. Both sets must coexist and stay distinguishable,
    which is why (patient, slice, source_file, annotator) — not just
    (patient, slice) — is the identity of an annotation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from medimageforge.logging_utils import get_logger
from medimageforge.manifest import connect

log = get_logger(__name__)

ANNOTATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS label_taxonomy (
    code         TEXT PRIMARY KEY,   -- 'subdural'
    display_name TEXT NOT NULL,      -- 'Subdural'
    category     TEXT NOT NULL,      -- hemorrhage_type | other_finding
    description  TEXT NOT NULL,
    source_column TEXT NOT NULL      -- which CSV column it came from
);

CREATE TABLE IF NOT EXISTS slice_annotations (
    id            INTEGER PRIMARY KEY,
    patient_id    TEXT NOT NULL,
    slice_no      INTEGER NOT NULL,
    mask_rel_path TEXT,              -- curated mask, NULL when none exists
    source_file   TEXT NOT NULL,     -- provenance: which file
    source_sha256 TEXT NOT NULL,     -- provenance: which VERSION of it
    annotator     TEXT NOT NULL,     -- provenance: who
    annotated_at  TEXT NOT NULL,     -- provenance: when we recorded it
    UNIQUE (patient_id, slice_no, source_file, annotator)
);

CREATE TABLE IF NOT EXISTS slice_labels (
    annotation_id INTEGER NOT NULL REFERENCES slice_annotations(id) ON DELETE CASCADE,
    label_code    TEXT NOT NULL REFERENCES label_taxonomy(code),
    value         INTEGER NOT NULL CHECK (value IN (0, 1)),
    PRIMARY KEY (annotation_id, label_code)
);

CREATE INDEX IF NOT EXISTS idx_ann_slice ON slice_annotations(patient_id, slice_no);
"""

# The taxonomy. Note what is here and what is not: five hemorrhage types and
# fracture are independent findings (74 slices have a fracture and no
# hemorrhage). `No_Hemorrhage` is absent on purpose — it is derived.
TAXONOMY: list[dict] = [
    {
        "code": "intraventricular",
        "display_name": "Intraventricular",
        "category": "hemorrhage_type",
        "description": "Bleeding into the ventricular system.",
        "source_column": "Intraventricular",
    },
    {
        "code": "intraparenchymal",
        "display_name": "Intraparenchymal",
        "category": "hemorrhage_type",
        "description": "Bleeding within the brain tissue itself.",
        "source_column": "Intraparenchymal",
    },
    {
        "code": "subarachnoid",
        "display_name": "Subarachnoid",
        "category": "hemorrhage_type",
        "description": "Bleeding into the subarachnoid space.",
        "source_column": "Subarachnoid",
    },
    {
        "code": "epidural",
        "display_name": "Epidural",
        "category": "hemorrhage_type",
        "description": "Bleeding between the skull and the dura mater.",
        "source_column": "Epidural",
    },
    {
        "code": "subdural",
        "display_name": "Subdural",
        "category": "hemorrhage_type",
        "description": "Bleeding between the dura mater and the arachnoid.",
        "source_column": "Subdural",
    },
    {
        "code": "fracture",
        "display_name": "Skull fracture",
        "category": "other_finding",
        "description": "Skull fracture present. Independent of hemorrhage.",
        "source_column": "Fracture_Yes_No",
    },
]

HEMORRHAGE_TYPES = [t["code"] for t in TAXONOMY if t["category"] == "hemorrhage_type"]

# From the dataset README: each slice was annotated by two radiologists who
# reached consensus. That is the provenance we record — not "unknown".
DEFAULT_ANNOTATOR = "radiologist-consensus"


@dataclass
class LoadReport:
    annotations: int = 0
    labels: int = 0
    with_mask: int = 0
    positive_slices: int = 0


def install_taxonomy(conn) -> None:
    """Insert (or refresh) the controlled vocabulary."""
    for term in TAXONOMY:
        conn.execute(
            """INSERT INTO label_taxonomy
                   (code, display_name, category, description, source_column)
               VALUES (:code, :display_name, :category, :description, :source_column)
               ON CONFLICT(code) DO UPDATE SET
                   display_name=excluded.display_name,
                   category=excluded.category,
                   description=excluded.description,
                   source_column=excluded.source_column""",
            term,
        )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _curated_mask_paths(conn) -> dict[tuple[str, int], str]:
    """Map (patient, slice) -> best available mask path, via the manifest.

    Prefers the CURATED mask (the true-binary PNG from Step 5) because that is
    what downstream code should consume, and falls back to the raw path.

    The curation table may not exist at all — the label store must work after
    `ingest` alone, without requiring `curate` to have run. We therefore probe
    for the table instead of assuming it, so Step 7 has no hidden dependency
    on Step 5.
    """
    has_curation = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='curation'"
    ).fetchone()

    if has_curation:
        query = """SELECT f.patient_id, f.slice_no, f.rel_path, c.curated_path
                   FROM files f
                   LEFT JOIN curation c ON c.rel_path = f.rel_path
                   WHERE f.kind = 'mask' AND f.status != 'missing'"""
    else:
        query = """SELECT patient_id, slice_no, rel_path, NULL AS curated_path
                   FROM files
                   WHERE kind = 'mask' AND status != 'missing'"""

    return {
        (r["patient_id"], r["slice_no"]): (r["curated_path"] or r["rel_path"])
        for r in conn.execute(query).fetchall()
    }


def load_annotations(
    db_path: Path,
    labels: pd.DataFrame,
    labels_csv: Path,
    patient_id_width: int,
    annotator: str = DEFAULT_ANNOTATOR,
) -> LoadReport:
    """Translate the wide CSV into the normalized label store. Idempotent.

    One CSV row becomes: one `slice_annotations` row (identity + provenance +
    mask reference) plus one `slice_labels` row per taxonomy term.
    """
    report = LoadReport()
    now = datetime.now(timezone.utc).isoformat()
    source_file = labels_csv.name
    source_sha = _sha256_file(labels_csv)

    with connect(db_path) as conn:
        conn.executescript(ANNOTATION_SCHEMA)
        install_taxonomy(conn)
        mask_paths = _curated_mask_paths(conn)

        for row in labels.itertuples(index=False):
            patient_id = f"{int(row.PatientNumber):0{patient_id_width}d}"
            slice_no = int(row.SliceNumber)
            mask_rel = mask_paths.get((patient_id, slice_no))

            conn.execute(
                """INSERT INTO slice_annotations
                       (patient_id, slice_no, mask_rel_path, source_file,
                        source_sha256, annotator, annotated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(patient_id, slice_no, source_file, annotator)
                   DO UPDATE SET mask_rel_path=excluded.mask_rel_path,
                                 source_sha256=excluded.source_sha256,
                                 annotated_at=excluded.annotated_at""",
                (patient_id, slice_no, mask_rel, source_file, source_sha, annotator, now),
            )
            annotation_id = conn.execute(
                """SELECT id FROM slice_annotations
                   WHERE patient_id=? AND slice_no=? AND source_file=? AND annotator=?""",
                (patient_id, slice_no, source_file, annotator),
            ).fetchone()["id"]

            positive = 0
            for term in TAXONOMY:
                value = int(getattr(row, term["source_column"]))
                if term["category"] == "hemorrhage_type":
                    positive += value
                conn.execute(
                    """INSERT INTO slice_labels (annotation_id, label_code, value)
                       VALUES (?, ?, ?)
                       ON CONFLICT(annotation_id, label_code)
                       DO UPDATE SET value=excluded.value""",
                    (annotation_id, term["code"], value),
                )
                report.labels += 1

            report.annotations += 1
            report.with_mask += 1 if mask_rel else 0
            report.positive_slices += 1 if positive else 0

        conn.commit()
    return report


def get_slice_annotation(
    db_path: Path, patient_id: str, slice_no: int, annotator: str | None = None
) -> list[dict]:
    """The Step 7 'Done' criterion: labels + mask path + provenance in ONE call.

    Returns one entry per annotator (today just the radiologist consensus; in
    Step 12 a model's suggestions will appear alongside it, which is exactly
    why this returns a list rather than a single record).
    """
    with connect(db_path) as conn:
        conn.executescript(ANNOTATION_SCHEMA)
        query = """SELECT a.id, a.patient_id, a.slice_no, a.mask_rel_path,
                          a.source_file, a.source_sha256, a.annotator, a.annotated_at
                   FROM slice_annotations a
                   WHERE a.patient_id = ? AND a.slice_no = ?"""
        params: list = [patient_id, slice_no]
        if annotator:
            query += " AND a.annotator = ?"
            params.append(annotator)

        results = []
        for ann in conn.execute(query, params).fetchall():
            rows = conn.execute(
                """SELECT l.label_code, l.value, t.display_name, t.category
                   FROM slice_labels l
                   JOIN label_taxonomy t ON t.code = l.label_code
                   WHERE l.annotation_id = ?
                   ORDER BY t.category, t.code""",
                (ann["id"],),
            ).fetchall()

            labels = {r["label_code"]: r["value"] for r in rows}
            positive = [r["label_code"] for r in rows if r["value"] == 1]
            hemorrhage = [c for c in positive if c in HEMORRHAGE_TYPES]
            results.append(
                {
                    "patient_id": ann["patient_id"],
                    "slice_no": ann["slice_no"],
                    "labels": labels,
                    "positive_labels": positive,
                    "hemorrhage_types": hemorrhage,
                    # Derived, never stored — see the module docstring.
                    "no_hemorrhage": not hemorrhage,
                    "mask_rel_path": ann["mask_rel_path"],
                    "provenance": {
                        "source_file": ann["source_file"],
                        "source_sha256": ann["source_sha256"],
                        "annotator": ann["annotator"],
                        "annotated_at": ann["annotated_at"],
                    },
                }
            )
    return results


def label_distribution(db_path: Path) -> list[dict]:
    """Positive count per label — reads the store, not the CSV."""
    with connect(db_path) as conn:
        conn.executescript(ANNOTATION_SCHEMA)
        return [
            dict(r)
            for r in conn.execute(
                """SELECT t.code, t.display_name, t.category,
                          SUM(l.value) AS positives, COUNT(*) AS total
                   FROM label_taxonomy t
                   LEFT JOIN slice_labels l ON l.label_code = t.code
                   GROUP BY t.code ORDER BY t.category, t.code"""
            )
        ]
