"""Error analysis: turning one aggregate number into actionable findings.

Why this module exists:
    Step 10 produced a single headline: test AUROC 0.768. That number cannot
    tell us *which* hemorrhages the model misses, *whether* it was measured on
    enough data to mean anything, or *which patient* is carrying the result.
    Aggregate metrics lie by omission. This module interrogates them.

    It is deliberately READ-ONLY with respect to the model: it loads an
    existing run's saved predictions and the decision threshold that run
    already chose on validation. Nothing here is tuned on test — re-tuning
    during "analysis" is how a test split quietly becomes a training signal.

Five questions it answers, and why each matters:

    1. Which subtypes can we even measure?  Support per hemorrhage type in
       the test split. If a type has 0 examples, the model's ability on it is
       UNKNOWN, not good — and reporting an overall number hides that.
    2. What does the model get wrong, by name?  Patient + slice number of the
       most confident mistakes, so a human can go look at the pixels.
    3. Is the score driven by one patient?  Leave-one-patient-out AUROC.
    4. Does the answer change at the patient level?  A radiologist reads a
       scan, not a slice: aggregating slices to a per-scan decision is the
       clinically meaningful unit.
    5. Where should the threshold sit?  A missed bleed costs far more than a
       false alarm, so the high-sensitivity operating point is reported
       alongside the F1-optimal one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from medimageforge.logging_utils import get_logger
from medimageforge.metrics import (
    auroc,
    bootstrap_auroc_ci,
    classification_metrics,
    trivial_baseline,
)

log = get_logger(__name__)

# The five hemorrhage subtypes, as named in the release index (Step 9) and the
# label taxonomy (Step 7). `fracture` is a separate finding, not a bleed type.
SUBTYPES = (
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
)

# Below this many positive examples a per-subtype metric is not worth quoting:
# with 1 case, recall is either 0.0 or 1.0 and carries no information.
MIN_SUPPORT = 2


def find_latest_run(runs_dir: Path) -> Path:
    """Most recent run directory. Run ids are timestamps, so sorting works."""
    candidates = sorted(p for p in runs_dir.iterdir() if (p / "run.json").is_file())
    if not candidates:
        raise FileNotFoundError(f"No runs with a run.json under {runs_dir}")
    return candidates[-1]


def load_run(run_dir: Path, split: str = "test") -> dict:
    """Load a run record plus its saved predictions for one split.

    Returns the record, and a tidy frame of one row per slice. The threshold
    comes FROM THE RECORD — this module never picks its own.
    """
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    arrays = np.load(run_dir / "predictions.npz", allow_pickle=True)

    prefix = "test" if split == "test" else "val"
    frame = pd.DataFrame(
        {
            "patient": arrays[f"{prefix}_patients"].astype(str),
            "slice_no": arrays[f"{prefix}_slices"].astype(int),
            "label": arrays[f"{prefix}_labels"].astype(int),
            "score": arrays[f"{prefix}_scores"].astype(float),
        }
    )
    return {
        "record": record,
        "predictions": frame,
        "threshold": float(record["selected_threshold"]),
        "run_id": record["run_id"],
        "dataset_version": record["dataset_version"],
        "split": split,
    }


def join_subtypes(predictions: pd.DataFrame, release_dir: Path, split: str) -> pd.DataFrame:
    """Attach per-subtype labels from the release index.

    The join key is (patient, slice_no) — which is exactly why slice numbers
    had to be added to the saved predictions. Relying on row order instead
    would couple this analysis to the loader's internal iteration order.
    """
    index = pd.read_csv(release_dir / "index.csv")
    index = index[(index["split"] == split) & (index["window"] == "brain")]
    columns = ["patient", "slice_no", *[c for c in SUBTYPES if c in index.columns]]
    if "fracture" in index.columns:
        columns.append("fracture")

    merged = predictions.merge(index[columns], on=["patient", "slice_no"], how="left")
    if len(merged) != len(predictions):
        raise ValueError(
            "Joining predictions to the release index changed the row count — "
            "duplicate (patient, slice_no) keys?"
        )
    missing = merged[[c for c in SUBTYPES if c in merged.columns]].isna().any(axis=1).sum()
    if missing:
        log.warning("%d predictions had no matching release row", missing)
    return merged


# ---------------------------------------------------------------------------
# 1. what can we measure?
# ---------------------------------------------------------------------------

def per_subtype_recall(
    frame: pd.DataFrame, threshold: float, min_support: int = MIN_SUPPORT
) -> list[dict]:
    """Recall for each hemorrhage subtype, with an honest measurability flag.

    Why recall and not AUROC per subtype: the model is a binary detector, so
    for a subtype the only sensible question is "of the slices that contain
    this kind of bleed, how many did it flag?".

    Why `measurable` exists: on v1.0 the test split contains ZERO subdural
    slices despite 56 in train. Quoting "recall 0.00" there would be a lie by
    formatting — the correct statement is that we cannot measure it at all.
    """
    results = []
    for subtype in SUBTYPES:
        if subtype not in frame.columns:
            continue
        subset = frame[frame[subtype] == 1]
        support = int(len(subset))
        measurable = support >= min_support
        detected = int((subset["score"] >= threshold).sum()) if support else 0
        results.append(
            {
                "subtype": subtype,
                "support": support,
                "detected": detected,
                "recall": round(detected / support, 4) if support else None,
                "measurable": measurable,
                "note": (
                    "no examples in this split — ability is UNKNOWN, not zero"
                    if support == 0
                    else (
                        f"only {support} example(s); recall can only be 0 or 1"
                        if not measurable
                        else ""
                    )
                ),
            }
        )
    return sorted(results, key=lambda r: -r["support"])


# ---------------------------------------------------------------------------
# 2. what does it get wrong, by name?
# ---------------------------------------------------------------------------

def hardest_cases(frame: pd.DataFrame, threshold: float, top_n: int = 8) -> dict:
    """The most confident mistakes, named by patient and slice.

    "Confident" means far on the wrong side of the threshold: a false negative
    scored 0.02 is a worse failure than one scored 0.45, because the model was
    not even close to hesitating. These are the cases worth a human's time —
    and the queue Step 12's active-learning loop will consume.
    """
    positives = frame[frame["label"] == 1]
    negatives = frame[frame["label"] == 0]

    false_negatives = positives[positives["score"] < threshold].nsmallest(top_n, "score")
    false_positives = negatives[negatives["score"] >= threshold].nlargest(top_n, "score")

    def rows(subset: pd.DataFrame, kind: str) -> list[dict]:
        return [
            {
                "kind": kind,
                "patient": row.patient,
                "slice_no": int(row.slice_no),
                "label": int(row.label),
                "score": round(float(row.score), 4),
                "subtypes": [s for s in SUBTYPES if s in frame.columns and getattr(row, s, 0) == 1],
            }
            for row in subset.itertuples(index=False)
        ]

    return {
        "false_negatives": rows(false_negatives, "false_negative"),
        "false_positives": rows(false_positives, "false_positive"),
    }


# ---------------------------------------------------------------------------
# 3. is one patient carrying the result?
# ---------------------------------------------------------------------------

def patient_contributions(frame: pd.DataFrame, threshold: float) -> list[dict]:
    """Per-patient share of the positives AND per-patient detection rate.

    Why detection rate belongs here rather than in a covariate analysis:
    measured on v1.0, per-patient recall ranges from 0.12 to 1.00. Performance
    is clustered by patient, which means any slice-level explanation ("large
    lesions are missed", "peripheral bleeds are missed") is confounded with
    patient identity when only 5 patients carry positives. Reporting the
    cluster structure keeps that confounding visible instead of inviting a
    tidy but unsupported story.
    """
    total_positive = int((frame["label"] == 1).sum())
    rows = []
    for patient, group in frame.groupby("patient"):
        positives = group[group["label"] == 1]
        n_pos = int(len(positives))
        detected = int((positives["score"] >= threshold).sum()) if n_pos else 0
        rows.append(
            {
                "patient": patient,
                "slices": int(len(group)),
                "positive_slices": n_pos,
                "detected": detected,
                "detection_rate": round(detected / n_pos, 4) if n_pos else None,
                "median_score": round(float(group["score"].median()), 4),
                "share_of_all_positives": (
                    round(n_pos / total_positive, 4) if total_positive else 0.0
                ),
            }
        )
    return sorted(rows, key=lambda r: -r["positive_slices"])


def leave_one_patient_out(frame: pd.DataFrame) -> list[dict]:
    """Recompute AUROC with each patient removed, one at a time.

    A stable metric barely moves. A large swing names the patient the headline
    number actually depends on — which, with 12 test patients, is a real risk
    rather than a theoretical one.
    """
    overall = auroc(frame["label"].to_numpy(), frame["score"].to_numpy())
    results = []
    for patient in sorted(frame["patient"].unique()):
        remaining = frame[frame["patient"] != patient]
        labels = remaining["label"].to_numpy()
        if len(np.unique(labels)) < 2:
            results.append(
                {
                    "excluded_patient": patient,
                    "auroc_without": None,
                    "delta": None,
                    "note": "removing this patient leaves a single class",
                }
            )
            continue
        without = auroc(labels, remaining["score"].to_numpy())
        results.append(
            {
                "excluded_patient": patient,
                "auroc_without": round(without, 4),
                "delta": round(without - overall, 4),
                "note": "",
            }
        )
    return sorted(
        results, key=lambda r: -abs(r["delta"]) if r["delta"] is not None else 0
    )


# ---------------------------------------------------------------------------
# 4. does the answer change at the patient level?
# ---------------------------------------------------------------------------

def aggregate_to_patients(
    frame: pd.DataFrame, rule: str = "max", top_k: int = 3
) -> pd.DataFrame:
    """Collapse slice predictions into one prediction per scan.

    Rules:
      max     — the scan is suspicious if ANY slice is. This mirrors how a
                radiologist reads a study and how a triage tool must behave:
                one convincing slice is enough to escalate.
      topk    — mean of the k highest slice scores. Less sensitive to a single
                noisy slice, at the cost of missing a bleed visible on only
                one slice.

    A patient's label is positive if any of its slices is positive.
    """
    rows = []
    for patient, group in frame.groupby("patient"):
        scores = np.sort(group["score"].to_numpy())[::-1]
        if rule == "max":
            score = float(scores[0])
        elif rule == "topk":
            score = float(scores[: min(top_k, len(scores))].mean())
        else:
            raise ValueError(f"unknown aggregation rule: {rule!r}")
        rows.append(
            {
                "patient": patient,
                "label": int(group["label"].max()),
                "score": score,
                "slices": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values("patient").reset_index(drop=True)


def patient_level_report(
    frame: pd.DataFrame,
    threshold: float,
    rule: str = "max",
    bootstrap_resamples: int = 2000,
    seed: int = 0,
) -> dict:
    """Patient-level metrics, with the granularity of the estimate stated.

    Why granularity is reported: AUROC over n_pos x n_neg patient pairs can
    only take multiples of 1/(n_pos*n_neg). With 5 positive and 7 negative
    patients that is 1/35 ~ 0.029 — so differences smaller than one step are
    not merely uncertain, they are unrepresentable.
    """
    patients = aggregate_to_patients(frame, rule=rule)
    labels = patients["label"].to_numpy()
    scores = patients["score"].to_numpy()
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    metrics = classification_metrics(labels, scores, threshold)
    metrics["auroc_ci"] = bootstrap_auroc_ci(
        labels, scores, patients["patient"].to_numpy(),
        n_resamples=bootstrap_resamples, seed=seed,
    )
    return {
        "rule": rule,
        "n_patients": int(len(patients)),
        "n_positive_patients": n_pos,
        "n_negative_patients": n_neg,
        "comparable_pairs": n_pos * n_neg,
        "auroc_granularity": round(1 / (n_pos * n_neg), 4) if n_pos and n_neg else None,
        "metrics": metrics,
        "per_patient": patients.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# 5. where should the threshold sit?
# ---------------------------------------------------------------------------

def threshold_sweep(
    frame: pd.DataFrame, target_recall: float = 0.90, n_points: int = 41
) -> dict:
    """Operating points across the score range, plus a high-sensitivity choice.

    In a stroke workup a missed hemorrhage can be fatal while a false alarm
    costs a radiologist a few seconds. The F1-optimal threshold balances the
    two as if they were equally bad; they are not. So we also report the
    lowest threshold that still reaches `target_recall`, and state what
    precision that costs, letting a clinician make the trade explicitly.
    """
    labels = frame["label"].to_numpy()
    scores = frame["score"].to_numpy()
    grid = np.linspace(0.0, 1.0, n_points)

    points = []
    for threshold in grid:
        m = classification_metrics(labels, scores, float(threshold))
        points.append(
            {
                "threshold": round(float(threshold), 4),
                "recall": round(m["recall"], 4),
                "precision": round(m["precision"], 4),
                "specificity": round(m["specificity"], 4),
                "f1": round(m["f1"], 4),
                "fp": int(m["fp"]),
                "fn": int(m["fn"]),
            }
        )

    # Highest threshold that still achieves the target recall: the strictest
    # (fewest false alarms) point meeting the safety requirement.
    achieving = [p for p in points if p["recall"] >= target_recall]
    high_sensitivity = max(achieving, key=lambda p: p["threshold"]) if achieving else None
    return {
        "target_recall": target_recall,
        "high_sensitivity_point": high_sensitivity,
        "achievable": high_sensitivity is not None,
        "points": points,
    }


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def evaluate_run(
    run_dir: Path,
    release_dir: Path,
    split: str = "test",
    top_n: int = 8,
    target_recall: float = 0.90,
    bootstrap_resamples: int = 2000,
) -> dict:
    """Run every analysis and return one report dict."""
    run = load_run(run_dir, split=split)
    frame = join_subtypes(run["predictions"], release_dir, split)
    threshold = run["threshold"]

    labels = frame["label"].to_numpy()
    scores = frame["score"].to_numpy()
    slice_metrics = classification_metrics(labels, scores, threshold)
    slice_metrics["auroc_ci"] = bootstrap_auroc_ci(
        labels, scores, frame["patient"].to_numpy(),
        n_resamples=bootstrap_resamples, seed=0,
    )

    return {
        "run_id": run["run_id"],
        "dataset_version": run["dataset_version"],
        "split": split,
        "threshold": threshold,
        "threshold_source": "selected on validation by the training run",
        "slice_level": slice_metrics,
        "baseline": trivial_baseline(labels),
        "per_subtype": per_subtype_recall(frame, threshold),
        "patient_level": {
            "max": patient_level_report(
                frame, threshold, "max", bootstrap_resamples
            ),
            "topk": patient_level_report(
                frame, threshold, "topk", bootstrap_resamples
            ),
        },
        "patient_contributions": patient_contributions(frame, threshold),
        "leave_one_patient_out": leave_one_patient_out(frame),
        "threshold_sweep": threshold_sweep(frame, target_recall),
        "hardest_cases": hardest_cases(frame, threshold, top_n),
    }


def render_markdown(report: dict) -> str:
    """A human-readable evaluation report."""
    slice_m = report["slice_level"]
    ci = slice_m["auroc_ci"]
    lines = [
        f"# Evaluation — {report['run_id']}",
        "",
        f"- dataset version: `{report['dataset_version']}`",
        f"- split: **{report['split']}**",
        f"- threshold: **{report['threshold']:.3f}** ({report['threshold_source']})",
        "",
        "## Slice level",
        "",
        "| metric | baseline | model |",
        "|---|---|---|",
    ]
    baseline = report["baseline"]
    for key in ("auroc", "balanced_accuracy", "recall", "precision", "f1", "accuracy"):
        lines.append(f"| {key} | {baseline[key]:.4f} | {slice_m[key]:.4f} |")
    lines += [
        "",
        f"AUROC {ci['auroc']:.4f}, 95% CI [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}] "
        f"resampling {ci['n_units']} patients.",
        "",
        f"Confusion: tp={slice_m['tp']:.0f} fp={slice_m['fp']:.0f} "
        f"tn={slice_m['tn']:.0f} fn={slice_m['fn']:.0f}",
        "",
        "## Per subtype — what can actually be measured",
        "",
        "| subtype | support | detected | recall | measurable | note |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["per_subtype"]:
        recall = "n/a" if row["recall"] is None else f"{row['recall']:.4f}"
        lines.append(
            f"| {row['subtype']} | {row['support']} | {row['detected']} | {recall} | "
            f"{'yes' if row['measurable'] else '**NO**'} | {row['note']} |"
        )

    lines += ["", "## Patient level (a radiologist reads a scan, not a slice)", ""]
    for rule in ("max", "topk"):
        block = report["patient_level"][rule]
        m = block["metrics"]
        lines.append(
            f"- **{rule}** rule: {block['n_positive_patients']} positive / "
            f"{block['n_negative_patients']} negative patients, "
            f"AUROC {m['auroc']:.4f} "
            f"(95% CI [{m['auroc_ci']['ci_low']:.3f}, {m['auroc_ci']['ci_high']:.3f}]), "
            f"recall {m['recall']:.4f}, precision {m['precision']:.4f}"
        )
    granularity = report["patient_level"]["max"]["auroc_granularity"]
    if granularity:
        lines.append(
            f"- Only {report['patient_level']['max']['comparable_pairs']} comparable "
            f"pairs exist, so patient AUROC moves in steps of {granularity:.3f}. "
            "Smaller differences are unrepresentable, not just uncertain."
        )

    lines += ["", "## Per-patient concentration and detection rate", "",
              "| patient | slices | positive | detected | recall | share of all positives |",
              "|---|---|---|---|---|---|"]
    for row in report["patient_contributions"]:
        if row["positive_slices"]:
            lines.append(
                f"| {row['patient']} | {row['slices']} | {row['positive_slices']} | "
                f"{row['detected']} | {row['detection_rate']:.2f} | "
                f"{row['share_of_all_positives']:.1%} |"
            )
    lines += [
        "",
        "Per-patient recall varies widely, so performance is clustered by patient. "
        "Any slice-level explanation of the failures is confounded with patient "
        "identity while so few patients carry positives.",
    ]

    lines += ["", "## Leave-one-patient-out AUROC", "",
              "| excluded patient | AUROC without | delta |", "|---|---|---|"]
    for row in report["leave_one_patient_out"][:6]:
        if row["auroc_without"] is None:
            lines.append(f"| {row['excluded_patient']} | n/a | {row['note']} |")
        else:
            lines.append(
                f"| {row['excluded_patient']} | {row['auroc_without']:.4f} | "
                f"{row['delta']:+.4f} |"
            )

    sweep = report["threshold_sweep"]
    lines += ["", "## Operating point", ""]
    if sweep["achievable"]:
        point = sweep["high_sensitivity_point"]
        lines.append(
            f"To reach recall >= {sweep['target_recall']:.2f} the threshold must drop to "
            f"**{point['threshold']:.3f}**, giving recall {point['recall']:.3f} at "
            f"precision {point['precision']:.3f} "
            f"({point['fp']} false positives, {point['fn']} missed)."
        )
    else:
        lines.append(
            f"Recall >= {sweep['target_recall']:.2f} is NOT achievable at any threshold "
            "for this model."
        )

    lines += ["", "## Hardest cases (named, for human review)", "",
              "| kind | patient | slice | score | subtypes |", "|---|---|---|---|---|"]
    for row in report["hardest_cases"]["false_negatives"]:
        lines.append(
            f"| missed | {row['patient']} | {row['slice_no']} | {row['score']:.4f} | "
            f"{', '.join(row['subtypes']) or '-'} |"
        )
    for row in report["hardest_cases"]["false_positives"]:
        lines.append(
            f"| false alarm | {row['patient']} | {row['slice_no']} | {row['score']:.4f} | - |"
        )
    lines.append("")
    return "\n".join(lines)


def render_hardest_cases(
    cases: dict,
    pseudonym_map: dict[str, str],
    curated_dir: Path,
    out_dir: Path,
) -> list[Path]:
    """Write an image per hardest case: the slice, and the mask where one exists.

    Why bother, when we already have the numbers: a table says the model missed
    patient X slice 14. Only the pixels say *why* — a faint bleed, an unusual
    window, a slice at the edge of the skull. This closes the loop with Step 3,
    where we first learned to look at the data.

    The filename carries patient, slice and score so the images are
    self-describing without needing to render text into them. Pseudonyms are
    resolved through the manifest, so these artifacts stay inside the
    controlled zone and never contain a real patient number.
    """
    from PIL import Image

    from medimageforge.imaging import load_gray, mask_from_jpeg, overlay_mask, side_by_side

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for kind, rows in (
        ("fn", cases["false_negatives"]),
        ("fp", cases["false_positives"]),
    ):
        for row in rows:
            real_id = pseudonym_map.get(row["patient"])
            if real_id is None:
                log.warning("cannot resolve %s — skipping image", row["patient"])
                continue
            slice_path = curated_dir / real_id / "brain" / f"{row['slice_no']}.png"
            if not slice_path.is_file():
                log.warning("missing curated slice %s", slice_path)
                continue

            image = load_gray(slice_path)
            mask_path = curated_dir / real_id / "brain" / f"{row['slice_no']}_mask.png"
            if mask_path.is_file():
                # Curated masks are already binary (Step 5), but thresholding
                # again is harmless and keeps this independent of that detail.
                panels = side_by_side(image, overlay_mask(image, mask_from_jpeg(load_gray(mask_path))))
            else:
                panels = side_by_side(image, image)

            name = (
                f"{kind}_{row['patient']}_slice{row['slice_no']:03d}"
                f"_score{row['score']:.3f}.png"
            )
            path = out_dir / name
            Image.fromarray(panels).save(path)
            written.append(path)
    return written


def write_report(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "evaluation.json"
    md_path = out_dir / "evaluation.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
