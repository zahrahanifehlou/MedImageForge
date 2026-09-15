"""Quality control: a dataset must earn its way to training.

Why this module exists:
    Steps 4-7 established what we HAVE. QC asks whether it is
    internally consistent enough to train on. Without gates, a data lake
    becomes a data swamp: labels pointing at missing images, masks that
    contradict their labels, the same patient in both train and test.

Severity, reusing the Step 5 vocabulary:
    ERROR   — would corrupt training or invalidate results. Fails the gate.
    WARNING — real, noteworthy, but the data stays usable. Gate still passes.

The leakage check is the interesting one — see `find_leakage` for why it is
two-stage and how its threshold was calibrated against real data instead of
being guessed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from medimageforge.logging_utils import get_logger
from medimageforge.manifest import connect

log = get_logger(__name__)

ERROR = "ERROR"
WARNING = "WARNING"


@dataclass
class QCReport:
    checks: list[dict] = field(default_factory=list)

    def add(
        self,
        name: str,
        detail: str,
        errors: list | None = None,
        warnings: list | None = None,
    ) -> None:
        errors = errors or []
        warnings = warnings or []
        self.checks.append(
            {
                "check": name,
                "result": "FAIL" if errors else ("WARN" if warnings else "PASS"),
                "detail": detail,
                "errors": errors,
                "warnings": warnings,
            }
        )

    @property
    def passed(self) -> bool:
        """Warnings do not fail the gate — only errors do."""
        return not any(c["errors"] for c in self.checks)

    @property
    def n_errors(self) -> int:
        return sum(len(c["errors"]) for c in self.checks)

    @property
    def n_warnings(self) -> int:
        return sum(len(c["warnings"]) for c in self.checks)

    def as_dict(self) -> dict:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "gate": "PASS" if self.passed else "FAIL",
            "n_errors": self.n_errors,
            "n_warnings": self.n_warnings,
            "checks": self.checks,
        }


# ---------------------------------------------------------------------------
# perceptual hashing
# ---------------------------------------------------------------------------

def dhash(path: Path, size: int = 8) -> np.ndarray:
    """Difference hash: 64 bits describing horizontal gradient structure.

    Resize to (size+1, size) and compare each pixel with its right neighbour.
    Robust to small intensity shifts and re-compression, which is exactly what
    an exact SHA-256 is not — two visually identical images saved at different
    JPEG qualities have different bytes but the same dhash.
    """
    im = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    array = np.asarray(im, dtype=np.int16)
    return (array[:, 1:] > array[:, :-1]).flatten()


def hamming_matrix(bits: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances for an (n, 64) boolean matrix.

    Encoding bits as +/-1 turns the dot product into (64 - 2*distance), so one
    matrix multiply replaces n^2 loops — 3.1M pairs in under a second.
    """
    signed = bits.astype(np.int8) * 2 - 1
    n_bits = bits.shape[1]
    return (n_bits - signed @ signed.T) // 2


def _normalized_vector(path: Path, size: int = 64) -> np.ndarray:
    """Downsampled, zero-mean unit-variance image — for correlation scoring."""
    im = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    vec = np.asarray(im, dtype=np.float32).flatten()
    return (vec - vec.mean()) / (vec.std() + 1e-8)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / a.size)


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------

def check_counts(conn) -> tuple[list, list]:
    """Per-patient slice counts must agree across windows.

    A brain slice with no bone counterpart is usable on its own (Step 5 warned
    about exactly this), so it is a WARNING — but an unexplained asymmetry is
    the kind of thing that must never pass silently.
    """
    per_patient: dict[str, dict[str, int]] = {}
    for row in conn.execute(
        """SELECT patient_id, window, COUNT(*) n FROM files
           WHERE kind='slice' AND status != 'missing'
           GROUP BY patient_id, window"""
    ):
        per_patient.setdefault(row["patient_id"], {})[row["window"]] = row["n"]

    warnings = []
    for patient_id, counts in sorted(per_patient.items()):
        brain, bone = counts.get("brain", 0), counts.get("bone", 0)
        if brain != bone:
            warnings.append(
                {
                    "patient_id": patient_id,
                    "issue": "window-count-mismatch",
                    "brain": brain,
                    "bone": bone,
                }
            )
    return [], warnings


def check_orphans(conn, labels: pd.DataFrame, patient_id_width: int) -> tuple[list, list]:
    """Cross-reference files against labels in both directions.

    A label with no image is unusable (ERROR). An image with no label cannot
    be trained on supervised (ERROR). A mask with no slice is an orphan
    annotation (ERROR).
    """
    slices = {
        (r["patient_id"], r["slice_no"])
        for r in conn.execute(
            "SELECT patient_id, slice_no FROM files"
            " WHERE kind='slice' AND window='brain' AND status != 'missing'"
        )
    }
    masks = {
        (r["patient_id"], r["slice_no"])
        for r in conn.execute(
            "SELECT patient_id, slice_no FROM files WHERE kind='mask' AND status != 'missing'"
        )
    }
    label_keys = {
        (f"{int(r.PatientNumber):0{patient_id_width}d}", int(r.SliceNumber))
        for r in labels.itertuples(index=False)
    }

    errors = []
    for patient_id, slice_no in sorted(label_keys - slices):
        errors.append(
            {"patient_id": patient_id, "slice_no": slice_no, "issue": "label-without-image"}
        )
    for patient_id, slice_no in sorted(slices - label_keys):
        errors.append(
            {"patient_id": patient_id, "slice_no": slice_no, "issue": "image-without-label"}
        )
    for patient_id, slice_no in sorted(masks - slices):
        errors.append(
            {"patient_id": patient_id, "slice_no": slice_no, "issue": "mask-without-slice"}
        )
    return errors, []


def check_mask_label_agreement(conn, labels: pd.DataFrame, patient_id_width: int) -> tuple[list, list]:
    """A mask means "hemorrhage is here" — the labels must say so too.

    Both directions are errors: a mask on a slice labelled hemorrhage-free is
    a contradiction, and a hemorrhage-positive slice with no delineation means
    a missing annotation.
    """
    types = ["Intraventricular", "Intraparenchymal", "Subarachnoid", "Epidural", "Subdural"]
    positive = {
        (f"{int(r.PatientNumber):0{patient_id_width}d}", int(r.SliceNumber))
        for r in labels.itertuples(index=False)
        if sum(int(getattr(r, t)) for t in types) > 0
    }
    masks = {
        (r["patient_id"], r["slice_no"])
        for r in conn.execute(
            "SELECT patient_id, slice_no FROM files WHERE kind='mask' AND status != 'missing'"
        )
    }

    errors = []
    for patient_id, slice_no in sorted(masks - positive):
        errors.append(
            {
                "patient_id": patient_id,
                "slice_no": slice_no,
                "issue": "mask-but-no-hemorrhage-label",
            }
        )
    for patient_id, slice_no in sorted(positive - masks):
        errors.append(
            {
                "patient_id": patient_id,
                "slice_no": slice_no,
                "issue": "hemorrhage-label-but-no-mask",
            }
        )
    return errors, []


def find_leakage(
    conn,
    data_dir: Path,
    max_hamming: int,
    min_correlation: float,
    top_n: int = 10,
) -> tuple[list, list, dict]:
    """Detect the same scan appearing under two different patient IDs.

    Why leakage matters
    -------------------
    Splitting by patient (Step 9) is worthless if the same scan exists twice
    under different IDs: it lands in train AND test, and the reported accuracy
    becomes fiction.

    Why this is TWO-stage
    ---------------------
    Stage 1 (cheap): dhash every slice and compare all ~3.1M pairs with one
    matrix multiply. Exact SHA-256 cannot do this job — a re-compressed copy
    has different bytes but identical appearance.

    Stage 2 (expensive): only for the handful of cross-patient candidates,
    load the real pixels and score correlation. Loading 2501 images to compare
    everything would be wasteful; loading ~100 is nothing.

    Why the threshold is CALIBRATED, not guessed
    --------------------------------------------
    Head CTs are anatomically stereotyped. Measured on this dataset:

        closest cross-patient candidate   correlation 0.950
        same-patient ADJACENT slices      correlation 0.896 median (max 0.994)

    The distributions overlap — different patients at the same anatomical
    level look more alike than consecutive slices of one patient. A threshold
    picked by intuition (say 0.9) would flag hundreds of distinct patients as
    duplicates. So the confirmation threshold sits above the entire
    cross-patient range, and we also report the top candidates for human
    review rather than pretending the gate is the whole answer.
    """
    rows = [
        dict(r)
        for r in conn.execute(
            """SELECT rel_path, patient_id, slice_no FROM files
               WHERE kind='slice' AND window='brain' AND status != 'missing'
               ORDER BY patient_id, slice_no"""
        )
    ]
    if len(rows) < 2:
        return [], [], {"candidates": 0, "compared": 0}

    bits = np.array([dhash(data_dir / r["rel_path"]) for r in rows])
    distances = hamming_matrix(bits)
    patients = np.array([r["patient_id"] for r in rows])

    upper = np.triu_indices(len(rows), k=1)
    is_cross = patients[upper[0]] != patients[upper[1]]
    candidate_idx = np.where(is_cross & (distances[upper] <= max_hamming))[0]

    # Stage 2: load pixels only for slices involved in a candidate pair.
    involved = sorted({upper[0][k] for k in candidate_idx} | {upper[1][k] for k in candidate_idx})
    vectors = {i: _normalized_vector(data_dir / rows[i]["rel_path"]) for i in involved}

    scored = []
    for k in candidate_idx:
        i, j = int(upper[0][k]), int(upper[1][k])
        score = correlation(vectors[i], vectors[j])
        scored.append(
            {
                "a": f"{rows[i]['patient_id']}/{rows[i]['slice_no']}",
                "b": f"{rows[j]['patient_id']}/{rows[j]['slice_no']}",
                "hamming": int(distances[i, j]),
                # Full precision is kept for the comparison and only rounded
                # for display: rounding BEFORE a threshold test would make
                # 0.999955 read as 1.0 and pass a 0.999999 gate.
                "correlation": round(score, 6),
                "_score": score,
            }
        )
    scored.sort(key=lambda s: -s["_score"])

    errors = [
        {k: v for k, v in pair.items() if k != "_score"}
        | {"issue": "cross-patient-near-duplicate"}
        for pair in scored
        if pair["_score"] >= min_correlation
    ]
    for pair in scored:
        pair.pop("_score", None)
    stats = {
        "candidates": len(scored),
        "compared": len(involved),
        "max_correlation": scored[0]["correlation"] if scored else None,
        "top_candidates": scored[:top_n],
    }
    return errors, [], stats


def check_demographics(
    conn, demographics: pd.DataFrame, labels: pd.DataFrame, patient_id_width: int
) -> tuple[list, list]:
    """Patient-level checks: coverage, plausibility, and label agreement.

    Note what is NOT used here: IQR outlier detection. Measured on this
    dataset, the IQR bounds are [-31.9, 83.1] years — they do not flag the
    1-day-old neonate at all, because the age distribution is very wide. A
    statistical rule is the wrong tool; a DOMAIN rule (paediatric age) is the
    right one. The neonate is genuine data, so it is a WARNING.
    """
    errors: list[dict] = []
    warnings: list[dict] = []

    on_disk = {
        r["patient_id"]
        for r in conn.execute(
            "SELECT DISTINCT patient_id FROM files WHERE patient_id IS NOT NULL"
        )
    }
    id_col = next(c for c in demographics.columns if c.lower().startswith("patient"))
    in_csv = {
        f"{int(v):0{patient_id_width}d}" for v in demographics[id_col].dropna()
    }
    for patient_id in sorted(on_disk - in_csv):
        errors.append({"patient_id": patient_id, "issue": "images-without-demographics"})
    for patient_id in sorted(in_csv - on_disk):
        errors.append({"patient_id": patient_id, "issue": "demographics-without-images"})

    age_col = next((c for c in demographics.columns if c.lower().startswith("age")), None)
    if age_col:
        ages = pd.to_numeric(demographics[age_col], errors="coerce")
        for idx, age in ages.items():
            patient_id = f"{int(demographics.loc[idx, id_col]):0{patient_id_width}d}"
            if pd.isna(age) or age < 0 or age > 120:
                errors.append(
                    {"patient_id": patient_id, "issue": "implausible-age", "value": None if pd.isna(age) else float(age)}
                )
            elif age < 1:
                warnings.append(
                    {
                        "patient_id": patient_id,
                        "issue": "paediatric-age-under-1-year",
                        "value": round(float(age), 4),
                    }
                )

    gender_col = next((c for c in demographics.columns if c.lower() == "gender"), None)
    if gender_col:
        unexpected = set(demographics[gender_col].dropna().unique()) - {"Male", "Female"}
        for value in sorted(unexpected):
            errors.append({"issue": "unexpected-gender-value", "value": str(value)})

    # Patient-level labels derived from the slices must match the
    # patient-level summary in demographics. Blank cells encode 0 there --
    # this check is what validates that assumption (0 disagreements).
    types = ["Intraventricular", "Intraparenchymal", "Subarachnoid", "Epidural", "Subdural"]
    shared = [t for t in types if t in demographics.columns]
    if shared:
        by_slice = labels.groupby("PatientNumber")[shared].max()
        by_patient = demographics.set_index(id_col)[shared].fillna(0)
        for patient in sorted(set(by_slice.index) & set(by_patient.index)):
            for kind in shared:
                if int(by_slice.loc[patient, kind]) != int(by_patient.loc[patient, kind]):
                    errors.append(
                        {
                            "patient_id": f"{int(patient):0{patient_id_width}d}",
                            "issue": "patient-vs-slice-label-disagreement",
                            "label": kind,
                        }
                    )
    return errors, warnings


def write_report(report: QCReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")


def run_qc(
    db_path: Path,
    data_dir: Path,
    labels: pd.DataFrame,
    demographics: pd.DataFrame,
    patient_id_width: int,
    max_hamming: int,
    min_correlation: float,
    skip_leakage: bool = False,
) -> tuple[QCReport, dict]:
    """Run every gate and return the report plus leakage statistics."""
    report = QCReport()
    leakage_stats: dict = {}

    with connect(db_path) as conn:
        errors, warnings = check_counts(conn)
        report.add("slice-counts", "per-patient brain/bone slice counts agree", errors, warnings)

        errors, warnings = check_orphans(conn, labels, patient_id_width)
        report.add("orphans", "labels <-> images <-> masks cross-reference", errors, warnings)

        errors, warnings = check_mask_label_agreement(conn, labels, patient_id_width)
        report.add(
            "mask-label-agreement",
            "a mask exists exactly when a hemorrhage is labelled",
            errors,
            warnings,
        )

        errors, warnings = check_demographics(conn, demographics, labels, patient_id_width)
        report.add(
            "demographics",
            "coverage, plausible values, patient-vs-slice label agreement",
            errors,
            warnings,
        )

        if skip_leakage:
            report.add("leakage", "skipped (--skip-leakage)", [], [])
        else:
            log.info("Perceptual-hashing slices for leakage detection ...")
            errors, warnings, leakage_stats = find_leakage(
                conn, data_dir, max_hamming, min_correlation
            )
            report.add(
                "leakage",
                f"two-stage: {leakage_stats.get('candidates', 0)} cross-patient dhash "
                f"candidates confirmed at correlation >= {min_correlation}",
                errors,
                warnings,
            )
    return report, leakage_stats
