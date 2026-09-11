"""Dataset explorer: measure the raw data before building pipelines for it.

Why this module exists:
    You cannot build a pipeline for data you have not measured. This module
    answers, with code instead of assumptions:

      - What is actually on disk?       (scan_raw_dir)
      - What do the labels claim?       (load_labels, load_demographics)
      - Do the files match what we      (verify_checksums)
        were given, byte for byte?

    The checksum part is the key lesson: a SHA-256 of every file is the
    dataset's fingerprint. Verifying it is the first job of any ingestion
    system — if bytes are corrupt or missing, every downstream result is
    meaningless, so we check *before* trusting anything.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from medimageforge.config import data_path
from medimageforge.logging_utils import get_logger

log = get_logger(__name__)

# Columns in hemorrhage_diagnosis.csv that are hemorrhage-type labels.
LABEL_COLUMNS = [
    "Intraventricular",
    "Intraparenchymal",
    "Subarachnoid",
    "Epidural",
    "Subdural",
    "No_Hemorrhage",
    "Fracture_Yes_No",
]


@dataclass
class PatientScan:
    """What we found on disk for one patient folder."""

    patient_id: str
    slice_counts: dict[str, int] = field(default_factory=dict)  # window -> n slices
    mask_count: int = 0

    @property
    def total_slices(self) -> int:
        return sum(self.slice_counts.values())


@dataclass
class ChecksumReport:
    """Result of comparing every file on disk against SHA256SUMS.txt."""

    verified: int = 0
    mismatched: list[str] = field(default_factory=list)  # hash differs
    missing: list[str] = field(default_factory=list)     # listed, absent on disk
    extra: list[str] = field(default_factory=list)       # on disk, not listed

    @property
    def ok(self) -> bool:
        return not (self.mismatched or self.missing)


def scan_raw_dir(raw_dir: Path, windows: list[str], mask_suffix: str) -> list[PatientScan]:
    """Walk data/Patients_CT and count slices and masks per patient/window.

    Rules encoded here (learned by looking at the data, not guessed):
      - patient folders are zero-padded numbers: 049, 050, ...
      - each has a brain/ and bone/ window folder
      - masks live in brain/ as <slice>_HGE_Seg.jpg and are NOT slices
    """
    patients: list[PatientScan] = []
    for patient_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        scan = PatientScan(patient_id=patient_dir.name)
        for window in windows:
            window_dir = patient_dir / window
            if not window_dir.is_dir():
                log.warning("Patient %s has no %s/ folder", scan.patient_id, window)
                scan.slice_counts[window] = 0
                continue
            files = [f.name for f in window_dir.iterdir() if f.suffix.lower() == ".jpg"]
            masks = [f for f in files if mask_suffix in f]
            scan.mask_count += len(masks)
            scan.slice_counts[window] = len(files) - len(masks)
        patients.append(scan)
    return patients


def load_labels(labels_csv: Path) -> pd.DataFrame:
    """Parse hemorrhage_diagnosis.csv (one row per patient+slice).

    encoding="utf-8-sig" strips the BOM; without it the first column would be
    named '\\ufeffPatientNumber' and silently break lookups.
    """
    df = pd.read_csv(labels_csv, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


def load_demographics(demographics_csv: Path) -> pd.DataFrame:
    """Parse patient_demographics.csv.

    Its header contains quoted multi-line names like "Age\\n(years)" —
    pandas handles the quoting; we just normalize column names.
    """
    df = pd.read_csv(demographics_csv, encoding="utf-8-sig")
    df.columns = [" ".join(c.split()) for c in df.columns]
    return df


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash a file in 1 MiB chunks — memory-flat regardless of file size."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksums(data_dir: Path, checksums_file: Path) -> ChecksumReport:
    """Compare every entry in SHA256SUMS.txt with the file on disk.

    The file is in `sha256sum` format: "<hash> <relative path>".
    Three failure modes, reported separately because they mean different things:
      mismatched — bytes changed (corruption, truncated download)
      missing    — listed but absent (incomplete copy)
      extra      — on disk but not listed (unexpected files we should notice)
    """
    report = ChecksumReport()
    listed: dict[str, str] = {}
    for line in checksums_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, rel_path = line.partition(" ")
        listed[rel_path.strip()] = digest

    for rel_path, expected in listed.items():
        path = data_dir / rel_path
        if not path.is_file():
            report.missing.append(rel_path)
        elif _sha256(path) != expected:
            report.mismatched.append(rel_path)
        else:
            report.verified += 1

    listed_set = set(listed)
    for path in data_dir.rglob("*"):
        if path.is_file():
            rel = path.relative_to(data_dir).as_posix()
            if rel not in listed_set:
                report.extra.append(rel)
    return report


def label_slice_exists(labels: pd.DataFrame, raw_dir: Path, patient_id_width: int) -> list[str]:
    """Find label rows whose brain-window slice is missing on disk.

    A label pointing at a file that does not exist is exactly the kind of
    inconsistency Step 8's quality gates will formalize — we surface it now.
    """
    missing = []
    for row in labels.itertuples(index=False):
        rel = f"{int(row.PatientNumber):0{patient_id_width}d}/brain/{int(row.SliceNumber)}.jpg"
        if not (raw_dir / rel).is_file():
            missing.append(rel)
    return missing
