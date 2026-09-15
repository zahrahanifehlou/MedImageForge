"""Privacy gate: prove there is no PHI, then pseudonymize working artifacts.

Why this module exists:
    Privacy is a GATE, not a report. Everything downstream (dataset releases,
    training, the API) must refuse to run if it has not passed. So this module
    returns a hard pass/fail, and the CLI exits non-zero on failure.

Two words that are constantly confused, and the difference matters legally:

    Anonymization   — irreversible. No key exists, so re-identification is
                      impossible for anyone, including us. Data ceases to be
                      personal data.
    Pseudonymization— reversible BY THE KEY HOLDER. Identifiers are replaced
                      with tokens; a secret allows authorized re-linking.
                      Still personal data, still regulated.

    Our JPEGs are anonymized at source (the dataset authors did it, and we
    verify it). Our patient *numbers* we pseudonymize: 049 -> PAT-xxxxxxxxxxxx.
    We keep the ability to re-link, because a platform that cannot answer
    "which patient was this?" for an authorized investigator is useless in
    a clinical setting.

The single most important implementation detail:
    A plain hash is NOT pseudonymization for a small identifier space.
    sha256("049") is a fixed, public value — an attacker hashes 000..999 and
    recovers every patient in under a millisecond. We therefore use a KEYED
    hash (HMAC-SHA256) with a secret salt that never enters the repository.
    The salt IS the re-identification key: protect it, and pseudonyms are
    opaque; leak it, and they are not.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image

from medimageforge.logging_utils import get_logger
from medimageforge.manifest import connect

log = get_logger(__name__)

PATIENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,   -- real ID: lives ONLY in the controlled zone
    pseudonym  TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);
"""

ENV_SALT = "MEDIMAGEFORGE_PSEUDONYM_SALT"

# Column names that would hold direct identifiers. Their mere presence is a
# finding — we should never have ingested such a column.
FORBIDDEN_COLUMN_TOKENS = (
    "name", "mrn", "medical_record", "dob", "birth", "address", "street",
    "city", "zip", "postal", "phone", "tel", "email", "ssn", "insurance",
    "accession", "physician", "doctor",
)

# Direct-identifier patterns for free-text values. Each maps to an identifier
# class named in the HIPAA Safe Harbor list.
TEXT_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone": re.compile(r"(?:\+?\d[\s.-]?){7,}\d"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "mrn": re.compile(r"\b(?:MRN|mrn)[\s:#]*\d+\b"),
    "date": re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}"
        r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b"
    ),
    "person_title": re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+"),
    "url": re.compile(r"https?://\S+"),
}

# HIPAA Safe Harbor: ages over 89 must be aggregated into a single 90+ group,
# because extreme ages are identifying in small populations.
MAX_SAFE_AGE = 89


@dataclass
class PrivacyReport:
    """Findings from the gate. `passed` is what downstream steps must check."""

    checks: list[dict] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str, findings: list | None = None) -> None:
        self.checks.append(
            {
                "check": name,
                "result": "PASS" if passed else "FAIL",
                "detail": detail,
                "findings": findings or [],
            }
        )

    @property
    def passed(self) -> bool:
        return all(c["result"] == "PASS" for c in self.checks)

    def as_dict(self) -> dict:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "gate": "PASS" if self.passed else "FAIL",
            "checks": self.checks,
        }


# ---------------------------------------------------------------------------
# The pseudonymization key
# ---------------------------------------------------------------------------

def load_or_create_salt(salt_file: Path) -> bytes:
    """Return the secret salt, creating one on first use.

    Precedence: the MEDIMAGEFORGE_PSEUDONYM_SALT env var wins (that is how a
    server or CI injects it), otherwise a local gitignored file. The file is
    written 0600 — readable only by its owner.

    This file is the re-identification key. It must never be committed, and
    losing it means every pseudonym becomes permanently unlinkable (which
    turns pseudonymization into de-facto anonymization).
    """
    env_value = os.environ.get(ENV_SALT)
    if env_value:
        return env_value.encode("utf-8")

    if salt_file.is_file():
        return salt_file.read_text(encoding="utf-8").strip().encode("utf-8")

    salt_file.parent.mkdir(parents=True, exist_ok=True)
    salt = secrets.token_hex(32)
    salt_file.write_text(salt, encoding="utf-8")
    salt_file.chmod(0o600)
    log.warning(
        "Generated a new pseudonymization salt at %s — back it up; "
        "without it existing pseudonyms cannot be re-linked.", salt_file
    )
    return salt.encode("utf-8")


def pseudonymize(patient_id: str, salt: bytes, length: int = 12) -> str:
    """Map a real patient ID to a stable, opaque token: 049 -> PAT-xxxxxxxxxxxx.

    HMAC-SHA256, not a plain hash: the identifier space here is tiny (82
    patients, 1000 possible 3-digit strings), so an unkeyed digest is trivially
    reversed by brute force. The secret salt is what makes the token opaque.

    Deterministic on purpose — the same patient always yields the same
    pseudonym, so pseudonymized artifacts can still be joined to each other.
    """
    digest = hmac.new(salt, patient_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"PAT-{digest[:length]}"


def read_pseudonym_map(db_path: Path) -> dict[str, str]:
    """Read the stored mapping (real id -> pseudonym). NEVER writes.

    Why this exists separately from `build_pseudonym_map`
    ----------------------------------------------------
    `build_pseudonym_map` persists what it computes. Anything that merely
    needs to *look up* pseudonyms must not call it: a reader that silently
    rewrites the table destroys its own evidence. Release verification did
    exactly that once — it recomputed the mapping and overwrote a corrupted
    table before checking it, so verification passed on broken data.

    Consumers (training, verification) read; only the privacy command writes.
    """
    with connect(db_path) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='patients'"
        ).fetchone()
        if not exists:
            return {}
        return {
            row["patient_id"]: row["pseudonym"]
            for row in conn.execute("SELECT patient_id, pseudonym FROM patients")
        }


def build_pseudonym_map(db_path: Path, salt: bytes) -> dict[str, str]:
    """Assign (and PERSIST) a pseudonym for every patient in the manifest.

    The mapping lives in the manifest DB — inside the controlled zone. That
    table is precisely the re-identification path, which is what makes this
    pseudonymization rather than anonymization.

    This function WRITES. Call it from the privacy gate, not from readers —
    use `read_pseudonym_map` to look pseudonyms up.
    """
    with connect(db_path) as conn:
        conn.executescript(PATIENTS_SCHEMA)
        patient_ids = [
            r["patient_id"]
            for r in conn.execute(
                "SELECT DISTINCT patient_id FROM files"
                " WHERE patient_id IS NOT NULL ORDER BY patient_id"
            )
        ]
        now = datetime.now(timezone.utc).isoformat()
        mapping = {}
        for pid in patient_ids:
            pseudo = pseudonymize(pid, salt)
            mapping[pid] = pseudo
            conn.execute(
                """INSERT INTO patients (patient_id, pseudonym, created_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(patient_id) DO UPDATE SET pseudonym=excluded.pseudonym""",
                (pid, pseudo, now),
            )
        conn.commit()
    return mapping


# ---------------------------------------------------------------------------
# PHI detection
# ---------------------------------------------------------------------------

def scan_image_metadata(paths: list[Path]) -> list[dict]:
    """Look for PHI hidden in image metadata rather than pixels.

    DICOM headers are the classic leak (PatientName, StudyDate...). These are
    JPEGs, but JPEG still carries EXIF and comment segments — a scanner export
    or editing tool can easily leave identifying text there.
    """
    findings = []
    for path in paths:
        with Image.open(path) as im:
            exif = im.getexif()
            if exif and len(exif):
                findings.append(
                    {"path": path.name, "issue": "exif-present", "keys": list(exif.keys())}
                )
            leftover = {
                k: str(v)[:80]
                for k, v in im.info.items()
                if k not in ("jfif", "jfif_version", "jfif_unit", "jfif_density", "dpi")
            }
            if leftover:
                findings.append(
                    {"path": path.name, "issue": "metadata-present", "keys": list(leftover)}
                )
    return findings


def scan_columns(df: pd.DataFrame, source: str) -> list[dict]:
    """Flag columns whose *name* implies a direct identifier."""
    findings = []
    for col in df.columns:
        normalized = re.sub(r"[^a-z]", "_", col.lower())
        for token in FORBIDDEN_COLUMN_TOKENS:
            if token in normalized:
                findings.append({"source": source, "column": col, "issue": f"name-implies:{token}"})
    return findings


def scan_free_text(df: pd.DataFrame, source: str) -> list[dict]:
    """Regex-scan every string cell for direct-identifier patterns."""
    findings = []
    for col in df.columns:
        series = df[col]
        if series.dtype != object:
            continue
        for idx, value in series.dropna().items():
            if not isinstance(value, str):
                continue
            for kind, pattern in TEXT_PATTERNS.items():
                match = pattern.search(value)
                if match:
                    findings.append(
                        {
                            "source": source,
                            "column": col,
                            "row": int(idx),
                            "issue": f"possible-{kind}",
                            "match": match.group(0)[:60],
                        }
                    )
    return findings


def check_age_rule(df: pd.DataFrame, source: str) -> list[dict]:
    """HIPAA Safe Harbor: ages above 89 must be grouped, not reported exactly."""
    findings = []
    for col in df.columns:
        if not col.lower().startswith("age"):
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        over = numeric[numeric > MAX_SAFE_AGE]
        for idx, value in over.items():
            findings.append(
                {
                    "source": source,
                    "column": col,
                    "row": int(idx),
                    "issue": f"age-over-{MAX_SAFE_AGE}",
                    "value": float(value),
                }
            )
    return findings


# ---------------------------------------------------------------------------
# De-identified export
# ---------------------------------------------------------------------------

def pseudonymize_frames(
    labels: pd.DataFrame,
    demographics: pd.DataFrame,
    mapping: dict[str, str],
    patient_id_width: int,
) -> list[dict]:
    """Build pseudonymized copies of the tabular data, in memory.

    Separate from writing so the gate can verify each frame while the real IDs
    are still available for the row-aligned leak check.

    The real ID column is DROPPED, not just supplemented — keeping it would
    make the whole exercise theatre.
    """

    def to_pseudonym(value) -> str:
        return mapping.get(f"{int(value):0{patient_id_width}d}", "PAT-UNKNOWN")

    labels_out = labels.copy()
    labels_out.insert(0, "patient", labels_out["PatientNumber"].map(to_pseudonym))
    real_label_ids = labels_out.pop("PatientNumber")

    demo_out = demographics.copy()
    id_col = next(c for c in demo_out.columns if c.lower().startswith("patient"))
    demo_out.insert(0, "patient", demo_out[id_col].map(to_pseudonym))
    real_demo_ids = demo_out.pop(id_col)

    return [
        {"name": "labels_pseudonymized.csv", "frame": labels_out, "real_ids": real_label_ids},
        {
            "name": "demographics_pseudonymized.csv",
            "frame": demo_out,
            "real_ids": real_demo_ids,
        },
    ]


def export_deidentified(items: list[dict], out_dir: Path) -> list[Path]:
    """Write the pseudonymized frames — the artifacts that may leave the zone."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in items:
        path = out_dir / item["name"]
        item["frame"].to_csv(path, index=False)
        paths.append(path)
    return paths


PSEUDONYM_RE = re.compile(r"PAT-[0-9a-f]{12}")


def _normalize_cell(value) -> str:
    """'49' / 49 / 49.0 all normalize to '49' so they compare equal."""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".")[0]
    return text


def find_id_leaks(
    frame: pd.DataFrame,
    real_ids: pd.Series,
    pseudonym_col: str = "patient",
    threshold: float = 0.2,
) -> list[dict]:
    """Verify an export cannot be used to recover the patient number.

    Why this is ROW-ALIGNED rather than a text search
    -------------------------------------------------
    The first version of this check scanned the exported file for the digits
    of every real patient ID. It failed: the demographics export contains an
    age of 49, and patient '049' exists. But that 49-year-old is a *different*
    patient — the numeric coincidence carries no identifying information.
    Patient IDs here run 49-130 and ages run 0-72, so the two ranges overlap
    and a blind text search cannot possibly distinguish them.

    The invariant we actually need is not "these digits appear nowhere" but
    "no column reproduces the patient identity FOR ITS OWN ROW". A leak means
    a column aligns with the real ID row by row; a coincidence does not. So we
    compare per row and flag a column only when it matches far more often than
    chance (`threshold`).
    """
    findings = []
    if pseudonym_col in frame.columns:
        malformed = [
            str(v)
            for v in frame[pseudonym_col]
            if not PSEUDONYM_RE.fullmatch(str(v))
        ]
        if malformed:
            findings.append(
                {
                    "column": pseudonym_col,
                    "issue": "malformed-pseudonym",
                    "examples": malformed[:5],
                }
            )
    else:
        findings.append({"column": pseudonym_col, "issue": "pseudonym-column-missing"})

    total = len(frame)
    for col in frame.columns:
        if col == pseudonym_col or total == 0:
            continue
        aligned = 0
        for value, pid in zip(frame[col], real_ids):
            cell = _normalize_cell(value)
            if not cell:
                continue
            # Compare numerically so the zero-padded folder form ('049') and
            # the bare CSV form (49) are both recognized as the same ID.
            if cell.isdigit() and str(pid).strip().isdigit():
                if int(cell) == int(str(pid).strip()):
                    aligned += 1
            elif cell == str(pid).strip():
                aligned += 1
        fraction = aligned / total
        if fraction > threshold:
            findings.append(
                {
                    "column": col,
                    "issue": "column-reproduces-patient-id",
                    "aligned_rows": aligned,
                    "fraction": round(fraction, 3),
                }
            )
    return findings


def write_report(report: PrivacyReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
