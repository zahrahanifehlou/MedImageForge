"""Dataset releases: immutable, reproducible, patient-level splits.

Why this module exists:
    "Which data trained this model?" must always be answerable. A folder path
    is not an answer — folders change. A *version* is an answer, so a release
    is an immutable snapshot with an index, a split, and a dataset card.

The single most important rule in medical imaging ML:
    SPLIT BY PATIENT, NEVER BY SLICE.
    A patient contributes ~30 slices of the same head, and adjacent slices are
    nearly identical (measured in Step 8: correlation 0.896 median). Splitting
    by slice puts slice 14 in train and slice 15 in test, so the model is
    tested on data it has effectively already seen. The score looks excellent
    and means nothing. `assert_no_patient_overlap` enforces the rule.

Why the release stores an INDEX rather than copies of the images:
    Copying 333 MB per version does not scale, and versions mostly overlap.
    Instead we record, for every included file, its curated path AND its
    sha256. That makes drift detectable: if a curated file is ever modified,
    verification fails and we know the release no longer describes reality.
    This is the same pointer-plus-hash idea DVC and git-annex use.

Privacy continuity (Step 6):
    A release is the artifact that LEAVES the controlled zone, so it is keyed
    by pseudonym (`PAT-...`), never by the real patient number. The real IDs
    stay in the manifest.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from medimageforge import __version__
from medimageforge.logging_utils import get_logger
from medimageforge.manifest import connect

log = get_logger(__name__)

SPLITS = ("train", "validation", "test")


@dataclass
class ReleaseSummary:
    version: str
    patients: dict[str, int] = field(default_factory=dict)
    slices: dict[str, int] = field(default_factory=dict)
    positive_slices: dict[str, int] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# the split
# ---------------------------------------------------------------------------

def _stable_hash(text: str) -> int:
    """A process-independent 32-bit hash of a string.

    Python's built-in hash() is randomized per process for str/bytes, so using
    it to derive a random seed silently breaks reproducibility across runs.
    """
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def patient_strata(
    labels: pd.DataFrame, patient_id_width: int
) -> dict[str, dict]:
    """Summarize each patient: the unit we actually split on.

    Slice-level labels are collapsed to patient level because the split
    operates on patients. `has_hemorrhage` is the stratification key.
    """
    types = ["Intraventricular", "Intraparenchymal", "Subarachnoid", "Epidural", "Subdural"]
    frame = labels.copy()
    frame["positive"] = frame[types].sum(axis=1) > 0

    summary: dict[str, dict] = {}
    for patient_number, group in frame.groupby("PatientNumber"):
        patient_id = f"{int(patient_number):0{patient_id_width}d}"
        summary[patient_id] = {
            "slices": int(len(group)),
            "positive_slices": int(group["positive"].sum()),
            "has_hemorrhage": bool(group["positive"].any()),
            "has_fracture": bool(group["Fracture_Yes_No"].max() > 0),
        }
    return summary


def split_patients(
    strata: dict[str, dict],
    ratios: dict[str, float],
    seed: int,
    stratify_by: str = "hemorrhage",
) -> dict[str, str]:
    """Assign every patient to exactly one split. Deterministic given `seed`.

    Ratios are applied WITHIN each stratum, so each split keeps roughly the
    same hemorrhage prevalence — important when only 36 of 82 patients are
    positive and the test split holds ~12 patients.

    Small strata degrade gracefully: with n=1, `round(1 * 0.70) == 1` sends
    the lone patient to train and leaves validation/test empty for that
    stratum. That is why we stratify on one factor only (see the config).
    """
    groups: dict[object, list[str]] = defaultdict(list)
    for patient_id, info in strata.items():
        key = info["has_hemorrhage"] if stratify_by == "hemorrhage" else info.get(stratify_by)
        groups[key].append(patient_id)

    assignment: dict[str, str] = {}
    # Sorting both the strata and the IDs makes the result independent of dict
    # ordering — reproducibility depends on it.
    for key in sorted(groups, key=str):
        patient_ids = sorted(groups[key])
        # NOT Python's hash(): for str it is salted per process (PYTHONHASHSEED),
        # so the same seed would produce a different split in a new process.
        # A cryptographic digest is stable across processes and machines.
        rng = np.random.default_rng([seed, _stable_hash(str(key))])
        shuffled = [patient_ids[i] for i in rng.permutation(len(patient_ids))]

        n = len(shuffled)
        first = round(n * ratios["train"])
        second = round(n * (ratios["train"] + ratios["validation"]))
        for patient_id in shuffled[:first]:
            assignment[patient_id] = "train"
        for patient_id in shuffled[first:second]:
            assignment[patient_id] = "validation"
        for patient_id in shuffled[second:]:
            assignment[patient_id] = "test"
    return assignment


def assert_no_patient_overlap(assignment: dict[str, str]) -> None:
    """The anti-leakage invariant, checked rather than assumed."""
    members = {split: set() for split in SPLITS}
    for patient_id, split in assignment.items():
        members[split].add(patient_id)
    for a in SPLITS:
        for b in SPLITS:
            if a < b and members[a] & members[b]:
                raise ValueError(
                    f"patient(s) in both {a} and {b}: {sorted(members[a] & members[b])}"
                )


# ---------------------------------------------------------------------------
# the release contents
# ---------------------------------------------------------------------------

def build_index(
    db_path: Path,
    assignment: dict[str, str],
    pseudonyms: dict[str, str],
    labels: pd.DataFrame,
    patient_id_width: int,
) -> pd.DataFrame:
    """One row per curated image in the release: pointer + hash + labels.

    Reads from the manifest and the curation table, so the release is derived
    from the platform's source of truth rather than from a directory listing.
    """
    types = ["Intraventricular", "Intraparenchymal", "Subarachnoid", "Epidural", "Subdural"]
    label_lookup = {
        (f"{int(r.PatientNumber):0{patient_id_width}d}", int(r.SliceNumber)): r
        for r in labels.itertuples(index=False)
    }

    with connect(db_path) as conn:
        rows = conn.execute(
            """SELECT f.patient_id, f.window, f.slice_no, f.kind,
                      c.curated_path, c.curated_sha256
               FROM files f
               JOIN curation c ON c.rel_path = f.rel_path
               WHERE c.decision = 'accepted' AND f.kind = 'slice'
               ORDER BY f.patient_id, f.window, f.slice_no"""
        ).fetchall()

    records = []
    for row in rows:
        patient_id = row["patient_id"]
        split = assignment.get(patient_id)
        if split is None:
            continue
        label_row = label_lookup.get((patient_id, row["slice_no"]))
        pseudonym = pseudonyms[patient_id]
        # The path is PSEUDONYMOUS. The curated file actually lives under the
        # real ID (curated/049/bone/1.png), but writing that here would put a
        # real patient number into an artifact designed to leave the
        # controlled zone — exactly what Step 6 forbids. Resolving
        # 'PAT-xxx/bone/1.png' back to a real location requires the manifest,
        # which is the re-identification control working as intended.
        record = {
            "patient": pseudonym,
            "split": split,
            "window": row["window"],
            "slice_no": row["slice_no"],
            "path": f"{pseudonym}/{row['window']}/{Path(row['curated_path']).name}",
            "sha256": row["curated_sha256"],
        }
        if label_row is not None:
            for kind in types:
                record[kind.lower()] = int(getattr(label_row, kind))
            record["fracture"] = int(label_row.Fracture_Yes_No)
            record["hemorrhage"] = int(sum(int(getattr(label_row, k)) for k in types) > 0)
        records.append(record)
    return pd.DataFrame(records)


def split_statistics(index: pd.DataFrame, strata: dict[str, dict], assignment: dict[str, str]) -> dict:
    """Counts per split, plus the balance we did NOT stratify on."""
    stats: dict[str, dict] = {}
    for split in SPLITS:
        patients = [p for p, s in assignment.items() if s == split]
        subset = index[index["split"] == split]
        brain = subset[subset["window"] == "brain"]
        stats[split] = {
            "patients": len(patients),
            "patients_with_hemorrhage": sum(
                1 for p in patients if strata[p]["has_hemorrhage"]
            ),
            "patients_with_fracture": sum(
                1 for p in patients if strata[p]["has_fracture"]
            ),
            "images": int(len(subset)),
            "brain_slices": int(len(brain)),
            "hemorrhage_slices": int(brain["hemorrhage"].sum()) if len(brain) else 0,
        }
        total = stats[split]["brain_slices"]
        stats[split]["hemorrhage_rate"] = (
            round(stats[split]["hemorrhage_slices"] / total, 4) if total else 0.0
        )
    return stats


def render_dataset_card(
    version: str,
    stats: dict,
    metadata: dict,
    known_issues: list[str],
) -> str:
    """A dataset card: what a future reader needs to use this release safely."""
    lines = [
        f"# Dataset card — {version}",
        "",
        f"Generated: {metadata['created_at']}",
        f"Produced by: medimageforge {metadata['code_version']}",
        "",
        "## Source",
        "",
        "Computed Tomography Images for Intracranial Hemorrhage Detection and",
        "Segmentation (Hssayeni et al., PhysioNet). Non-contrast head CT,",
        "5 mm slices, two window settings (brain, bone), 650x650 grayscale.",
        "",
        f"- labels file: `{metadata['labels_file']}`",
        f"- labels sha256: `{metadata['labels_sha256']}`",
        "",
        "## Contents",
        "",
        "| Split | Patients | w/ hemorrhage | w/ fracture | Images | Brain slices | Hemorrhage slices | Rate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for split in SPLITS:
        s = stats[split]
        lines.append(
            f"| {split} | {s['patients']} | {s['patients_with_hemorrhage']} | "
            f"{s['patients_with_fracture']} | {s['images']} | {s['brain_slices']} | "
            f"{s['hemorrhage_slices']} | {s['hemorrhage_rate']} |"
        )

    lines += [
        "",
        "## How the split was made",
        "",
        f"- **Unit: the patient.** Splitting by slice would leak: a patient's ~30",
        "  slices are near-identical, so the model would be tested on data it had",
        "  already seen. No patient appears in more than one split (enforced).",
        f"- Stratified by patient-level hemorrhage status.",
        f"- Ratios {metadata['ratios']} applied within each stratum.",
        f"- Deterministic: seed `{metadata['seed']}`. Re-running reproduces this split exactly.",
        "",
        "## Privacy",
        "",
        "Patients are identified by pseudonym (`PAT-...`) only. Real patient",
        "numbers remain inside the controlled zone (the manifest). Re-linking",
        "requires the pseudonymization salt, which is not part of this release.",
        "",
        "## Files in this release",
        "",
        "- `index.csv` — one row per image: pseudonym, split, window, slice,",
        "  curated path, sha256, and labels",
        "- `splits.csv` — pseudonym to split assignment",
        "- `metadata.json` — machine-readable provenance and configuration",
        "- `CHECKSUMS.txt` — sha256 of every file in this release",
        "- `dataset_card.md` — this document",
        "",
        "Images are referenced by path and hash, not copied: a release is a",
        "pointer set plus fingerprints, so drift in the curated zone is",
        "detectable. Verify with `python -m medimageforge release --verify`.",
        "",
        "## Known issues and limitations",
        "",
    ]
    lines += [f"- {issue}" for issue in known_issues]
    lines += [
        "",
        "## Intended use",
        "",
        "Research and learning. Not a clinical device. The test split contains",
        "~12 patients, so metrics have wide confidence intervals — report",
        "patient-level results and treat small differences as noise.",
        "",
    ]
    return "\n".join(lines)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_release(
    out_dir: Path,
    index: pd.DataFrame,
    assignment: dict[str, str],
    pseudonyms: dict[str, str],
    stats: dict,
    metadata: dict,
    known_issues: list[str],
) -> ReleaseSummary:
    """Write the snapshot, then make it read-only.

    Immutability is enforced two ways: the caller refuses to overwrite an
    existing version, and every file is written once then chmod 0444 so a
    later pipeline cannot silently modify a published release.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    index.to_csv(out_dir / "index.csv", index=False)
    pd.DataFrame(
        sorted(
            ({"patient": pseudonyms[p], "split": s} for p, s in assignment.items()),
            key=lambda r: (r["split"], r["patient"]),
        )
    ).to_csv(out_dir / "splits.csv", index=False)

    (out_dir / "metadata.json").write_text(
        json.dumps({**metadata, "statistics": stats}, indent=2), encoding="utf-8"
    )
    (out_dir / "dataset_card.md").write_text(
        render_dataset_card(metadata["version"], stats, metadata, known_issues),
        encoding="utf-8",
    )

    # Checksums of the release's own files — written last, and excluding itself.
    checksum_lines = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "CHECKSUMS.txt":
            checksum_lines.append(f"{_sha256_bytes(path.read_bytes())} {path.name}")
    (out_dir / "CHECKSUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    for path in out_dir.iterdir():
        if path.is_file():
            path.chmod(0o444)

    summary = ReleaseSummary(version=metadata["version"])
    summary.patients = {s: stats[s]["patients"] for s in SPLITS}
    summary.slices = {s: stats[s]["images"] for s in SPLITS}
    summary.positive_slices = {s: stats[s]["hemorrhage_slices"] for s in SPLITS}
    summary.files = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    return summary


def verify_release(
    release_dir: Path, curated_dir: Path, pseudonyms: dict[str, str] | None = None
) -> dict:
    """Check a published release against its own checksums and the curated zone.

    Three independent questions:
      - were the release's own files modified since publication?
      - do the images it points at still have the recorded content?
      - does the release leak a real patient ID?

    `pseudonyms` (real ID -> pseudonym) is needed to resolve the release's
    pseudonymous paths back to real curated files. Verification therefore only
    works from inside the controlled zone — which is the point.
    """
    findings: dict[str, list] = {
        "release_files": [],
        "referenced_images": [],
        "privacy": [],
    }
    reverse = {v: k for k, v in (pseudonyms or {}).items()}

    checksums = release_dir / "CHECKSUMS.txt"
    if not checksums.is_file():
        findings["release_files"].append({"issue": "missing-CHECKSUMS.txt"})
        return findings

    expected = {}
    for line in checksums.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, _, name = line.partition(" ")
            expected[name.strip()] = digest
    for name, digest in expected.items():
        path = release_dir / name
        if not path.is_file():
            findings["release_files"].append({"file": name, "issue": "missing"})
        elif _sha256_bytes(path.read_bytes()) != digest:
            findings["release_files"].append({"file": name, "issue": "modified"})

    index_path = release_dir / "index.csv"
    if index_path.is_file():
        index = pd.read_csv(index_path)

        # Privacy: no real patient number may appear anywhere in the release.
        for real_id in reverse.values():
            if any(
                index[column].astype(str).str.contains(rf"\b{real_id}\b", regex=True).any()
                for column in index.columns
            ):
                findings["privacy"].append(
                    {"issue": "real-patient-id-in-index", "value": real_id}
                )

        for row in index.itertuples(index=False):
            pseudonym, _, remainder = row.path.partition("/")
            real_id = reverse.get(pseudonym)
            if real_id is None:
                findings["referenced_images"].append(
                    {"file": row.path, "issue": "unresolvable-pseudonym"}
                )
                continue
            path = curated_dir / real_id / remainder
            if not path.is_file():
                findings["referenced_images"].append({"file": row.path, "issue": "missing"})
            elif _sha256_bytes(path.read_bytes()) != row.sha256:
                findings["referenced_images"].append(
                    {"file": row.path, "issue": "content-changed"}
                )
    return findings
