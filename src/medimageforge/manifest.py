"""The manifest: a SQLite registry of every file the platform knows about.

Why this module exists:
    The filesystem can store files, but it cannot answer questions:
    "which files belong to patient 049?", "what changed since last ingest?",
    "which slices passed curation?". A manifest is a table of record —
    every file gets one row holding its identity, content fingerprint,
    and lifecycle status. All later steps (curation, QC, versioning)
    read and update these rows instead of re-walking the filesystem.

    SQLite specifically: zero setup, one file, real SQL. The lesson —
    a database beats a CSV the moment you need queries and updates —
    transfers directly to RDS/DynamoDB in Phase 6.

Idempotency contract:
    Running ingest twice on unchanged data must change nothing except
    `last_seen` timestamps. Re-runnability is what makes a pipeline safe
    to retry after a crash — you never get duplicate or half-applied state.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id          INTEGER PRIMARY KEY,
    rel_path    TEXT NOT NULL UNIQUE,   -- path relative to data_dir; the file's stable identity
    kind        TEXT NOT NULL,          -- slice | mask | metadata
    patient_id  TEXT,                   -- '049' — NULL for metadata files
    window      TEXT,                   -- 'brain' | 'bone' — NULL for metadata
    slice_no    INTEGER,                -- 14 — NULL for metadata
    sha256      TEXT NOT NULL,          -- content fingerprint
    size_bytes  INTEGER NOT NULL,
    status      TEXT NOT NULL,          -- raw | missing (later: curated, rejected, ...)
    first_seen  TEXT NOT NULL,          -- ISO-8601 UTC
    last_seen   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_patient ON files(patient_id);
CREATE INDEX IF NOT EXISTS idx_files_kind    ON files(kind);
"""


@dataclass
class IngestReport:
    """What one ingest run did — the numbers that prove idempotency."""

    total_on_disk: int = 0
    new: int = 0
    unchanged: int = 0
    updated: int = 0      # path known but content changed
    missing: int = 0      # registered before, absent on disk now


def connect(db_path: Path) -> sqlite3.Connection:
    """Open (and create if needed) the manifest database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(rel_path: Path, windows: list[str], mask_suffix: str) -> dict:
    """Turn a path into manifest fields.

    'Patients_CT/049/brain/14.jpg'        -> slice, patient 049, brain, 14
    'Patients_CT/049/brain/14_HGE_Seg.jpg'-> mask,  patient 049, brain, 14
    'hemorrhage_diagnosis.csv'            -> metadata (no patient/window/slice)
    """
    parts = rel_path.parts
    record = {"kind": "metadata", "patient_id": None, "window": None, "slice_no": None}
    if len(parts) == 4 and parts[0] == "Patients_CT" and parts[2] in windows:
        stem = rel_path.stem
        record.update(patient_id=parts[1], window=parts[2])
        if mask_suffix in stem:
            record["kind"] = "mask"
            record["slice_no"] = int(stem.split("_")[0])
        else:
            record["kind"] = "slice"
            record["slice_no"] = int(stem)
    return record


def ingest(
    data_dir: Path,
    db_path: Path,
    windows: list[str],
    mask_suffix: str,
) -> IngestReport:
    """Register every file under data_dir in the manifest. Safe to re-run.

    For each file on disk we upsert by rel_path:
      absent row            -> insert,           counted as new
      same sha256 + size    -> touch last_seen,  counted as unchanged
      different fingerprint -> update in place,  counted as updated

    Rows whose files vanished from disk are marked status='missing'
    rather than deleted — a manifest records history, it doesn't rewrite it.
    """
    report = IngestReport()
    now = datetime.now(timezone.utc).isoformat()

    with connect(db_path) as conn:
        known = {row["rel_path"]: row for row in conn.execute("SELECT * FROM files")}
        seen: set[str] = set()

        for path in sorted(data_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(data_dir).as_posix()
            seen.add(rel)
            report.total_on_disk += 1

            digest = _sha256(path)
            size = path.stat().st_size
            existing = known.get(rel)

            if existing is None:
                fields = classify(Path(rel), windows, mask_suffix)
                conn.execute(
                    """INSERT INTO files
                       (rel_path, kind, patient_id, window, slice_no,
                        sha256, size_bytes, status, first_seen, last_seen)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 'raw', ?, ?)""",
                    (rel, fields["kind"], fields["patient_id"], fields["window"],
                     fields["slice_no"], digest, size, now, now),
                )
                report.new += 1
            elif existing["sha256"] == digest and existing["size_bytes"] == size:
                conn.execute(
                    "UPDATE files SET last_seen = ?, status = 'raw' WHERE rel_path = ?",
                    (now, rel),
                )
                report.unchanged += 1
            else:
                conn.execute(
                    """UPDATE files SET sha256 = ?, size_bytes = ?,
                       status = 'raw', last_seen = ? WHERE rel_path = ?""",
                    (digest, size, now, rel),
                )
                report.updated += 1

        for rel in set(known) - seen:
            conn.execute(
                "UPDATE files SET status = 'missing', last_seen = ? WHERE rel_path = ?",
                (now, rel),
            )
            report.missing += 1

        conn.commit()
    return report
