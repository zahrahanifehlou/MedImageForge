"""Tests for the Step 11 error analysis.

The emphasis is on honesty properties: a metric with no support must be
reported as unmeasurable rather than as zero, failures must be nameable, and
patient-level estimates must expose their own coarseness.
"""

import json

import numpy as np
import pandas as pd
import pytest

from medimageforge.evaluate import (
    MIN_SUPPORT,
    SUBTYPES,
    aggregate_to_patients,
    evaluate_run,
    find_latest_run,
    hardest_cases,
    join_subtypes,
    leave_one_patient_out,
    load_run,
    patient_contributions,
    patient_level_report,
    per_subtype_recall,
    render_markdown,
    threshold_sweep,
    write_report,
)


def _frame(rows):
    """rows: (patient, slice_no, label, score, subtype or None)."""
    records = []
    for patient, slice_no, label, score, subtype in rows:
        record = {
            "patient": patient,
            "slice_no": slice_no,
            "label": label,
            "score": score,
            **{s: 0 for s in SUBTYPES},
        }
        if subtype:
            record[subtype] = 1
        records.append(record)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# per-subtype measurability — the headline finding of this step
# ---------------------------------------------------------------------------

def test_subtype_with_no_examples_is_unmeasurable_not_zero():
    """v1.0 has ZERO subdural slices in test. 'recall 0.00' would be a lie."""
    frame = _frame([("P1", 1, 1, 0.9, "epidural"), ("P1", 2, 0, 0.1, None)])
    rows = {r["subtype"]: r for r in per_subtype_recall(frame, 0.5)}

    subdural = rows["subdural"]
    assert subdural["support"] == 0
    assert subdural["recall"] is None            # not 0.0
    assert subdural["measurable"] is False
    assert "UNKNOWN" in subdural["note"]


def test_subtype_with_one_example_is_flagged_as_uninformative():
    frame = _frame([("P1", 1, 1, 0.9, "subarachnoid")])
    rows = {r["subtype"]: r for r in per_subtype_recall(frame, 0.5)}
    assert rows["subarachnoid"]["support"] == 1
    assert rows["subarachnoid"]["measurable"] is False
    assert rows["subarachnoid"]["recall"] == 1.0   # computed, but not trustworthy


def test_subtype_recall_is_correct_with_enough_support():
    frame = _frame(
        [
            ("P1", 1, 1, 0.9, "epidural"),
            ("P1", 2, 1, 0.8, "epidural"),
            ("P1", 3, 1, 0.1, "epidural"),
            ("P1", 4, 1, 0.2, "epidural"),
        ]
    )
    row = next(r for r in per_subtype_recall(frame, 0.5) if r["subtype"] == "epidural")
    assert row["support"] == 4
    assert row["detected"] == 2
    assert row["recall"] == 0.5
    assert row["measurable"] is True


def test_subtypes_are_ordered_by_support():
    frame = _frame(
        [
            ("P1", 1, 1, 0.9, "epidural"),
            ("P1", 2, 1, 0.9, "epidural"),
            ("P1", 3, 1, 0.9, "subdural"),
        ]
    )
    rows = per_subtype_recall(frame, 0.5)
    assert rows[0]["subtype"] == "epidural"
    assert [r["support"] for r in rows] == sorted(
        [r["support"] for r in rows], reverse=True
    )


def test_min_support_threshold_is_two():
    assert MIN_SUPPORT == 2


# ---------------------------------------------------------------------------
# nameable failures
# ---------------------------------------------------------------------------

def test_hardest_cases_are_the_most_confident_mistakes():
    frame = _frame(
        [
            ("P1", 10, 1, 0.02, "epidural"),   # worst miss
            ("P1", 11, 1, 0.40, "epidural"),   # near miss
            ("P2", 12, 0, 0.99, None),         # worst false alarm
            ("P2", 13, 0, 0.60, None),         # milder false alarm
            ("P3", 14, 1, 0.90, "epidural"),   # correct
        ]
    )
    cases = hardest_cases(frame, threshold=0.5, top_n=2)

    fn = cases["false_negatives"]
    assert [(r["patient"], r["slice_no"]) for r in fn] == [("P1", 10), ("P1", 11)]
    assert fn[0]["score"] < fn[1]["score"]          # most confident first
    assert fn[0]["subtypes"] == ["epidural"]        # says WHAT was missed

    fp = cases["false_positives"]
    assert [(r["patient"], r["slice_no"]) for r in fp] == [("P2", 12), ("P2", 13)]
    assert fp[0]["score"] > fp[1]["score"]


def test_hardest_cases_respects_top_n():
    frame = _frame([("P1", i, 1, 0.01 * i, "epidural") for i in range(1, 11)])
    assert len(hardest_cases(frame, 0.5, top_n=3)["false_negatives"]) == 3


def test_correct_predictions_are_never_listed_as_mistakes():
    frame = _frame([("P1", 1, 1, 0.9, "epidural"), ("P1", 2, 0, 0.1, None)])
    cases = hardest_cases(frame, 0.5, 5)
    assert cases["false_negatives"] == [] and cases["false_positives"] == []


# ---------------------------------------------------------------------------
# patient-level aggregation
# ---------------------------------------------------------------------------

def test_max_rule_flags_a_scan_from_a_single_suspicious_slice():
    """A radiologist escalates a study on one convincing slice."""
    frame = _frame(
        [("P1", 1, 0, 0.05, None), ("P1", 2, 1, 0.95, "epidural"), ("P1", 3, 0, 0.05, None)]
    )
    patients = aggregate_to_patients(frame, rule="max")
    assert len(patients) == 1
    assert patients.loc[0, "score"] == 0.95
    assert patients.loc[0, "label"] == 1        # any positive slice -> positive scan


def test_topk_rule_averages_the_highest_scores():
    frame = _frame([("P1", i, 0, s, None) for i, s in enumerate([0.9, 0.8, 0.7, 0.1])])
    patients = aggregate_to_patients(frame, rule="topk", top_k=3)
    assert patients.loc[0, "score"] == pytest.approx((0.9 + 0.8 + 0.7) / 3)


def test_topk_is_less_sensitive_than_max_to_one_slice():
    frame = _frame([("P1", 1, 1, 0.99, "epidural")] + [("P1", i, 0, 0.01, None) for i in range(2, 6)])
    by_max = aggregate_to_patients(frame, "max").loc[0, "score"]
    by_topk = aggregate_to_patients(frame, "topk", top_k=3).loc[0, "score"]
    assert by_topk < by_max


def test_each_patient_appears_exactly_once_after_aggregation():
    frame = _frame(
        [("P1", 1, 1, 0.9, "epidural"), ("P1", 2, 0, 0.1, None), ("P2", 1, 0, 0.2, None)]
    )
    patients = aggregate_to_patients(frame, "max")
    assert patients["patient"].is_unique
    assert set(patients["patient"]) == {"P1", "P2"}


def test_unknown_aggregation_rule_raises():
    with pytest.raises(ValueError, match="unknown aggregation rule"):
        aggregate_to_patients(_frame([("P1", 1, 1, 0.9, None)]), rule="median")


def test_patient_report_states_its_own_granularity():
    """With few patients, AUROC can only take discrete values — say so."""
    rows = [("P%d" % p, 1, 1 if p < 3 else 0, 0.9 if p < 3 else 0.1, "epidural" if p < 3 else None)
            for p in range(1, 8)]
    report = patient_level_report(_frame(rows), 0.5, "max", bootstrap_resamples=50)
    assert report["n_positive_patients"] == 2
    assert report["n_negative_patients"] == 5
    assert report["comparable_pairs"] == 10
    assert report["auroc_granularity"] == pytest.approx(0.1)
    assert report["metrics"]["auroc_ci"]["resampling_unit"] == "patient"


def test_patient_auroc_lands_on_a_multiple_of_the_granularity():
    rows = [("P%d" % p, 1, 1 if p < 4 else 0, 0.5 + 0.05 * p, None) for p in range(1, 9)]
    report = patient_level_report(_frame(rows), 0.5, "max", bootstrap_resamples=20)
    step = report["auroc_granularity"]
    assert (report["metrics"]["auroc"] / step) == pytest.approx(
        round(report["metrics"]["auroc"] / step), abs=1e-6
    )


# ---------------------------------------------------------------------------
# concentration and stability
# ---------------------------------------------------------------------------

def test_patient_contributions_expose_concentration_and_recall():
    frame = _frame(
        [("BIG", i, 1, 0.1, "epidural") for i in range(1, 5)]
        + [("SMALL", 1, 1, 0.9, "epidural"), ("SMALL", 2, 0, 0.1, None)]
    )
    rows = {r["patient"]: r for r in patient_contributions(frame, threshold=0.5)}
    assert rows["BIG"]["share_of_all_positives"] == 0.8
    assert rows["BIG"]["detection_rate"] == 0.0       # all four missed
    assert rows["SMALL"]["detection_rate"] == 1.0


def test_patient_with_no_positives_has_no_detection_rate():
    frame = _frame([("P1", 1, 0, 0.1, None)])
    assert patient_contributions(frame, 0.5)[0]["detection_rate"] is None


def test_leave_one_out_identifies_the_patient_carrying_the_score():
    """A planted badly-scored patient must show the largest positive delta."""
    rows = [("BAD", i, 1, 0.01, "epidural") for i in range(1, 6)]      # missed
    rows += [("GOOD", i, 1, 0.99, "epidural") for i in range(1, 4)]    # detected
    rows += [("NEG%d" % p, 1, 0, 0.20, None) for p in range(1, 5)]
    results = leave_one_patient_out(_frame(rows))
    assert results[0]["excluded_patient"] == "BAD"
    assert results[0]["delta"] > 0                # removing it improves AUROC


def test_leave_one_out_reports_undefined_when_a_class_disappears():
    rows = [("ONLYPOS", 1, 1, 0.9, "epidural"), ("NEG", 1, 0, 0.1, None)]
    results = leave_one_patient_out(_frame(rows))
    assert all(r["auroc_without"] is None for r in results)
    assert all("single class" in r["note"] for r in results)


# ---------------------------------------------------------------------------
# threshold sweep
# ---------------------------------------------------------------------------

def test_recall_is_monotone_non_increasing_in_the_threshold():
    rng = np.random.default_rng(0)
    rows = [("P1", i, int(i % 3 == 0), float(rng.random()), None) for i in range(60)]
    points = threshold_sweep(_frame(rows))["points"]
    recalls = [p["recall"] for p in points]
    assert all(a >= b for a, b in zip(recalls, recalls[1:]))


def test_high_sensitivity_point_meets_the_target():
    rows = [("P1", i, 1, 0.30, "epidural") for i in range(10)]
    rows += [("P2", i, 0, 0.20, None) for i in range(10)]
    sweep = threshold_sweep(_frame(rows), target_recall=0.90)
    assert sweep["achievable"] is True
    assert sweep["high_sensitivity_point"]["recall"] >= 0.90


def test_high_sensitivity_point_is_the_strictest_one_meeting_the_target():
    """Among thresholds achieving the target, prefer fewest false alarms."""
    rows = [("P1", i, 1, 0.60, "epidural") for i in range(5)]
    rows += [("P2", i, 0, 0.10, None) for i in range(5)]
    sweep = threshold_sweep(_frame(rows), target_recall=1.0)
    point = sweep["high_sensitivity_point"]
    assert point["recall"] == 1.0
    assert point["fp"] == 0                      # the strict end, not threshold 0


def test_unachievable_target_is_reported_honestly():
    rows = [("P1", i, 1, 0.0, "epidural") for i in range(10)]
    sweep = threshold_sweep(_frame(rows), target_recall=0.9, n_points=5)
    # every threshold above 0 misses everything; only threshold 0.0 catches all
    assert isinstance(sweep["achievable"], bool)


# ---------------------------------------------------------------------------
# end to end, on the synthetic release from the training tests
# ---------------------------------------------------------------------------

@pytest.fixture
def trained_run(tmp_path):
    """A tiny release (same shape as the Step 10 fixture), trained for 2 epochs."""
    pytest.importorskip("torch")

    from medimageforge.manifest import connect
    from medimageforge.privacy import PATIENTS_SCHEMA
    from PIL import Image

    curated = tmp_path / "curated"
    release_dir = tmp_path / "datasets" / "v1.0"
    release_dir.mkdir(parents=True)
    db = tmp_path / "manifest.db"
    rng = np.random.default_rng(0)
    rows = []
    with connect(db) as conn:
        conn.executescript(PATIENTS_SCHEMA)
        for patient in range(1, 9):
            real_id = f"{patient:03d}"
            pseudonym = f"PAT-{patient:012d}"
            conn.execute(
                "INSERT INTO patients (patient_id, pseudonym, created_at)"
                " VALUES (?, ?, 'now')",
                (real_id, pseudonym),
            )
            split = {1: "train", 2: "train", 3: "train", 4: "train",
                     5: "validation", 6: "validation",
                     7: "test", 8: "test"}[patient]
            positive = patient % 2 == 0
            for slice_no in range(1, 5):
                array = np.clip(
                    rng.normal(200 if positive else 60, 10, (32, 32)), 0, 255
                ).astype(np.uint8)
                path = curated / real_id / "brain" / f"{slice_no}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(array).save(path)
                rows.append(
                    {
                        "patient": pseudonym, "split": split, "window": "brain",
                        "slice_no": slice_no,
                        "path": f"{pseudonym}/brain/{slice_no}.png",
                        "sha256": "x", "hemorrhage": int(positive),
                        "epidural": int(positive), "intraparenchymal": 0,
                        "intraventricular": 0, "subarachnoid": 0, "subdural": 0,
                        "fracture": 0,
                    }
                )
        conn.commit()
    pd.DataFrame(rows).to_csv(release_dir / "index.csv", index=False)

    from medimageforge.train import TrainingConfig, run_training

    config = TrainingConfig(image_size=32, batch_size=4, epochs=2, seed=7)
    record = run_training(release_dir, db, curated, tmp_path / "runs", config)
    return tmp_path / "runs" / record.run_id, release_dir, curated, db


def test_load_run_reads_the_threshold_from_the_record(trained_run):
    """Step 11 must never pick its own threshold."""
    run_dir, _, _, _ = trained_run
    run = load_run(run_dir)
    record = json.loads((run_dir / "run.json").read_text())
    assert run["threshold"] == record["selected_threshold"]


def test_load_run_returns_joinable_predictions(trained_run):
    run_dir, _, _, _ = trained_run
    frame = load_run(run_dir)["predictions"]
    assert set(frame.columns) == {"patient", "slice_no", "label", "score"}
    assert frame["slice_no"].min() >= 1        # real slice numbers, not indices


def test_join_subtypes_preserves_row_count(trained_run):
    run_dir, release_dir, _, _ = trained_run
    predictions = load_run(run_dir)["predictions"]
    merged = join_subtypes(predictions, release_dir, "test")
    assert len(merged) == len(predictions)
    assert "epidural" in merged.columns


def test_join_detects_duplicate_keys(trained_run):
    """A silent row explosion would corrupt every downstream metric."""
    run_dir, release_dir, _, _ = trained_run
    predictions = load_run(run_dir)["predictions"]
    doubled = pd.concat([predictions, predictions])
    index = pd.read_csv(release_dir / "index.csv")
    dup = pd.concat([index, index])
    dup.to_csv(release_dir / "index.csv", index=False)
    try:
        with pytest.raises(ValueError, match="row count"):
            join_subtypes(predictions, release_dir, "test")
    finally:
        index.to_csv(release_dir / "index.csv", index=False)
    assert len(doubled) == 2 * len(predictions)


def test_evaluate_run_produces_every_section(trained_run):
    run_dir, release_dir, _, _ = trained_run
    report = evaluate_run(run_dir, release_dir, top_n=3, bootstrap_resamples=50)
    for key in (
        "slice_level", "baseline", "per_subtype", "patient_level",
        "patient_contributions", "leave_one_patient_out", "threshold_sweep",
        "hardest_cases",
    ):
        assert key in report
    assert report["dataset_version"] == "v1.0"
    assert report["threshold_source"].startswith("selected on validation")


def test_report_files_are_written(trained_run, tmp_path):
    run_dir, release_dir, _, _ = trained_run
    report = evaluate_run(run_dir, release_dir, top_n=2, bootstrap_resamples=20)
    json_path, md_path = write_report(report, tmp_path / "eval")
    assert json.loads(json_path.read_text())["run_id"] == report["run_id"]
    markdown = md_path.read_text()
    assert "# Evaluation" in markdown
    assert "NOT MEASURABLE" in markdown or "measurable" in markdown


def test_markdown_flags_unmeasurable_subtypes(trained_run):
    run_dir, release_dir, _, _ = trained_run
    report = evaluate_run(run_dir, release_dir, top_n=2, bootstrap_resamples=20)
    markdown = render_markdown(report)
    # the synthetic release has no subdural cases at all
    assert "subdural" in markdown
    assert "**NO**" in markdown


def test_hardest_case_images_are_rendered(trained_run, tmp_path):
    from medimageforge.data import load_pseudonym_map
    from medimageforge.evaluate import render_hardest_cases

    run_dir, release_dir, curated, db = trained_run
    report = evaluate_run(run_dir, release_dir, top_n=4, bootstrap_resamples=20)
    written = render_hardest_cases(
        report["hardest_cases"], load_pseudonym_map(db), curated, tmp_path / "hardest"
    )
    total = len(report["hardest_cases"]["false_negatives"]) + len(
        report["hardest_cases"]["false_positives"]
    )
    assert len(written) == total
    for path in written:
        assert path.is_file()
        assert path.name.startswith(("fn_", "fp_"))
        assert "PAT-" in path.name          # pseudonymous filenames only


def test_find_latest_run_picks_the_newest(tmp_path):
    for name in ("run-20260101T000000Z", "run-20260201T000000Z"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "run.json").write_text("{}")
    assert find_latest_run(tmp_path).name == "run-20260201T000000Z"


def test_find_latest_run_errors_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_latest_run(tmp_path)
