"""Curation: turn validated raw files into a trusted, normalized curated zone.

Why this module exists:
    Raw data is not ML-ready. Before anything trains on it we must answer,
    per file: is it readable? the expected shape? a duplicate? does it have
    a label? Files that pass are written — normalized — into a SEPARATE
    curated zone. Files that fail are recorded with a reason and never
    silently dropped.

Two design rules that matter more than the code:

    1. Zone separation. We READ data/ and WRITE artifacts/curated/. Raw is
       never modified. Any curation bug is fixed by deleting the curated
       zone and re-running — raw is always the fallback.

    2. Reject vs warn. A file that cannot be used is REJECTED. A file that
       is usable but noteworthy (e.g. a brain slice whose bone counterpart
       is missing) gets a WARNING and is still curated. Conflating the two
       silently throws away good data.

Normalization, and why:
    JPEG is lossy — every re-save degrades the image, so the curated zone is
    PNG (lossless) and no downstream step ever re-compresses. Masks get
    thresholded to true 0/255, fixing the JPEG-ringing defect we found in
    Step 3 once, centrally, instead of in every consumer.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from medimageforge.logging_utils import get_logger
from medimageforge.manifest import connect

log = get_logger(__name__)

CURATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS curation (
    rel_path       TEXT PRIMARY KEY,   -- source file, relative to data_dir
    decision       TEXT NOT NULL,      -- accepted | rejected
    reasons        TEXT,               -- why rejected ('' when accepted)
    warnings       TEXT,               -- usable, but noteworthy
    curated_path   TEXT,               -- relative to curated_dir; NULL if rejected
    source_sha256  TEXT NOT NULL,      -- provenance: WHICH version of the raw file
    curated_sha256 TEXT,               -- fingerprint of what we produced
    curated_at     TEXT NOT NULL
);
"""


@dataclass
class CurationReport:
    """Outcome of one curation run."""

    accepted: int = 0
    rejected: int = 0
    skipped: int = 0  # already curated, source unchanged
    rejections: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "skipped": self.skipped,
            "rejections": self.rejections,
            "warnings": self.warnings,
        }


def validate_image(path: Path, expected_size: tuple[int, int]) -> tuple[list[str], np.ndarray | None]:
    """Check one image file. Returns (rejection_reasons, pixels or None).

    Rules, each mapping to a real failure mode:
      unreadable        -> truncated download / not actually an image
      wrong-dimensions  -> a model expecting 650x650 would crash or mis-crop
      blank             -> all-black or near-constant: no information content
    """
    reasons: list[str] = []
    try:
        with Image.open(path) as im:
            im.load()                      # force decode: catches truncated files
            if im.size != tuple(expected_size):
                reasons.append(f"wrong-dimensions:{im.size}")
            array = np.asarray(im.convert("L"))
    except Exception as exc:                # noqa: BLE001 — any decode failure is a rejection
        return [f"unreadable:{type(exc).__name__}"], None

    if array.max() == 0 or float(array.std()) < 1.0:
        reasons.append("blank-or-constant")
    return reasons, array


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_and_save(
    array: np.ndarray, out_path: Path, is_mask: bool, mask_threshold: int
) -> str:
    """Write the normalized image as lossless PNG; return its sha256.

    Slices: grayscale as-is. Masks: thresholded to a true binary 0/255 image.
    """
    if is_mask:
        array = np.where(array > mask_threshold, 255, 0).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # A 2-D uint8 array already implies mode "L" — passing mode= is deprecated.
    Image.fromarray(array).save(out_path, format="PNG", optimize=True)
    return _sha256_bytes(out_path.read_bytes())


def curated_rel_path(patient_id: str, window: str, slice_no: int, is_mask: bool) -> str:
    """Layout of the curated zone: predictable, flat, no JPEG anywhere.

    curated/049/brain/14.png       (slice)
    curated/049/brain/14_mask.png  (mask — '_mask' is clearer than '_HGE_Seg')
    """
    name = f"{slice_no}_mask.png" if is_mask else f"{slice_no}.png"
    return f"{patient_id}/{window}/{name}"


def curate(
    data_dir: Path,
    curated_dir: Path,
    db_path: Path,
    labels,
    expected_size: tuple[int, int],
    mask_threshold: int,
    force: bool = False,
) -> CurationReport:
    """Validate every registered image, then write normalized copies.

    Reads the file list from the manifest (Step 4) rather than re-walking the
    disk — that is the point of having a source of truth. Idempotent: files
    already curated from an unchanged source are skipped unless force=True.
    """
    report = CurationReport()
    now = datetime.now(timezone.utc).isoformat()
    label_keys = {
        (int(r.PatientNumber), int(r.SliceNumber)) for r in labels.itertuples(index=False)
    }

    with connect(db_path) as conn:
        conn.executescript(CURATION_SCHEMA)

        rows = [
            dict(r)
            for r in conn.execute(
                """SELECT rel_path, kind, patient_id, window, slice_no, sha256
                   FROM files
                   WHERE kind IN ('slice','mask') AND status != 'missing'
                   ORDER BY rel_path"""
            )
        ]
        already = {
            r["rel_path"]: r
            for r in conn.execute("SELECT rel_path, source_sha256 FROM curation")
        }

        # --- cross-file checks, computed once up front ---------------------
        # Exact duplicates: same content hash under different paths. We keep
        # the first path (sorted) and reject the rest — content is identical,
        # so nothing is lost, but we record that it happened.
        by_hash: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            by_hash[r["sha256"]].append(r["rel_path"])
        duplicate_of = {
            path: paths[0]
            for paths in by_hash.values()
            if len(paths) > 1
            for path in paths[1:]
        }
        # Window pairing: which (patient, slice) exist in each window.
        present: dict[str, set[tuple[str, int]]] = defaultdict(set)
        for r in rows:
            if r["kind"] == "slice":
                present[r["window"]].add((r["patient_id"], r["slice_no"]))

        for row in rows:
            rel = row["rel_path"]
            is_mask = row["kind"] == "mask"
            prior = already.get(rel)
            if prior and prior["source_sha256"] == row["sha256"] and not force:
                report.skipped += 1
                continue

            reasons: list[str] = []
            warnings: list[str] = []

            if rel in duplicate_of:
                reasons.append(f"duplicate-of:{duplicate_of[rel]}")

            img_reasons, array = validate_image(data_dir / rel, expected_size)
            reasons.extend(img_reasons)

            key = (int(row["patient_id"]), row["slice_no"])
            if not is_mask and key not in label_keys:
                # A slice with no label row cannot be used for training.
                reasons.append("no-label-row")
            if is_mask and (row["patient_id"], row["slice_no"]) not in present["brain"]:
                reasons.append("mask-without-slice")
            if not is_mask:
                other = "bone" if row["window"] == "brain" else "brain"
                if (row["patient_id"], row["slice_no"]) not in present[other]:
                    # Usable on its own — noted, not rejected.
                    warnings.append(f"missing-{other}-counterpart")

            if reasons or array is None:
                report.rejected += 1
                report.rejections.append({"rel_path": rel, "reasons": reasons})
                conn.execute(
                    """INSERT INTO curation (rel_path, decision, reasons, warnings,
                           curated_path, source_sha256, curated_sha256, curated_at)
                       VALUES (?, 'rejected', ?, ?, NULL, ?, NULL, ?)
                       ON CONFLICT(rel_path) DO UPDATE SET
                           decision='rejected', reasons=excluded.reasons,
                           warnings=excluded.warnings, curated_path=NULL,
                           source_sha256=excluded.source_sha256,
                           curated_sha256=NULL, curated_at=excluded.curated_at""",
                    (rel, "; ".join(reasons), "; ".join(warnings), row["sha256"], now),
                )
                conn.execute(
                    "UPDATE files SET status='rejected' WHERE rel_path = ?", (rel,)
                )
                continue

            out_rel = curated_rel_path(
                row["patient_id"], row["window"], row["slice_no"], is_mask
            )
            curated_sha = normalize_and_save(
                array, curated_dir / out_rel, is_mask, mask_threshold
            )
            report.accepted += 1
            if warnings:
                report.warnings.append({"rel_path": rel, "warnings": warnings})
            conn.execute(
                """INSERT INTO curation (rel_path, decision, reasons, warnings,
                       curated_path, source_sha256, curated_sha256, curated_at)
                   VALUES (?, 'accepted', '', ?, ?, ?, ?, ?)
                   ON CONFLICT(rel_path) DO UPDATE SET
                       decision='accepted', reasons='', warnings=excluded.warnings,
                       curated_path=excluded.curated_path,
                       source_sha256=excluded.source_sha256,
                       curated_sha256=excluded.curated_sha256,
                       curated_at=excluded.curated_at""",
                (rel, "; ".join(warnings), out_rel, row["sha256"], curated_sha, now),
            )
            conn.execute(
                "UPDATE files SET status='curated' WHERE rel_path = ?", (rel,)
            )

        conn.commit()
    return report


def curation_state(db_path: Path) -> dict:
    """Summarize the curated zone as it stands now, from the curation table.

    Why this is separate from CurationReport: a report of one *run* says only
    what changed that run. After an idempotent re-run it would show "0
    warnings" and a reviewer would wrongly conclude the dataset is spotless.
    The persisted report must describe the *state of the dataset*, so we read
    every recorded decision back out of the manifest.
    """
    with connect(db_path) as conn:
        conn.executescript(CURATION_SCHEMA)
        totals = {
            r["decision"]: r["n"]
            for r in conn.execute(
                "SELECT decision, COUNT(*) n FROM curation GROUP BY decision"
            )
        }
        rejections = [
            {"rel_path": r["rel_path"], "reasons": r["reasons"].split("; ")}
            for r in conn.execute(
                "SELECT rel_path, reasons FROM curation WHERE decision='rejected'"
                " ORDER BY rel_path"
            )
        ]
        warnings = [
            {"rel_path": r["rel_path"], "warnings": r["warnings"].split("; ")}
            for r in conn.execute(
                "SELECT rel_path, warnings FROM curation"
                " WHERE warnings != '' ORDER BY rel_path"
            )
        ]
    return {
        "accepted": totals.get("accepted", 0),
        "rejected": totals.get("rejected", 0),
        "rejections": rejections,
        "warnings": warnings,
    }


def write_report(report: CurationReport, state: dict, path: Path) -> None:
    """Persist both views as JSON — an artifact a reviewer can read later.

    'last_run' answers "what did I just do?"; 'dataset_state' answers
    "what is in the curated zone?". Both matter, and they are not the same.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "last_run": report.as_dict(),
        "dataset_state": state,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
