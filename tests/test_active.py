"""Tests for the Step 12 active-learning loop.

The properties that matter are experimental hygiene: identical evaluation data
across arms, a genuine random control, reproducible selection, and pool
bookkeeping that cannot leak a patient from unlabelled to labelled by accident.
"""

import numpy as np
import pandas as pd
import pytest

from medimageforge.active import (
    entropy,
    patient_uncertainty,
    restrict_train_patients,
    select_patients,
    stratified_pool_split,
    summarize,
    write_experiment_index,
)


# ---------------------------------------------------------------------------
# uncertainty
# ---------------------------------------------------------------------------

def test_entropy_is_maximal_at_the_decision_boundary():
    assert entropy(np.array([0.5]))[0] == pytest.approx(1.0)


def test_entropy_is_minimal_when_confident():
    values = entropy(np.array([0.01, 0.5, 0.99]))
    assert values[0] < values[1] and values[2] < values[1]
    assert values[0] == pytest.approx(values[2], abs=1e-9)  # symmetric


def test_entropy_is_symmetric_about_one_half():
    assert entropy(np.array([0.3]))[0] == pytest.approx(entropy(np.array([0.7]))[0])


def test_entropy_never_returns_nan_at_the_extremes():
    """p=0 and p=1 would give log(0); the clip must prevent NaN."""
    values = entropy(np.array([0.0, 1.0]))
    assert np.all(np.isfinite(values))


def test_entropy_ranking_matches_least_confidence():
    """For binary problems all the usual uncertainty measures agree in order."""
    scores = np.array([0.02, 0.45, 0.6, 0.95])
    by_entropy = np.argsort(-entropy(scores))
    by_margin = np.argsort(-(1 - np.abs(2 * scores - 1)))
    assert list(by_entropy) == list(by_margin)


def test_patient_uncertainty_mean_rule():
    frame = pd.DataFrame(
        {
            "patient": ["A", "A", "B", "B"],
            "slice_no": [1, 2, 1, 2],
            "score": [0.5, 0.5, 0.99, 0.01],   # A is ambiguous, B is confident
        }
    )
    ranked = patient_uncertainty(frame, rule="mean")
    assert list(ranked["patient"]) == ["A", "B"]      # most uncertain first
    assert ranked.loc[0, "uncertainty"] == pytest.approx(1.0)


def test_patient_uncertainty_max_rule_differs_from_mean():
    frame = pd.DataFrame(
        {
            "patient": ["A"] * 3 + ["B"] * 3,
            "slice_no": [1, 2, 3] * 2,
            # A: one maximally confusing slice. B: uniformly middling.
            "score": [0.5, 0.99, 0.99, 0.7, 0.7, 0.7],
        }
    )
    by_max = patient_uncertainty(frame, "max")
    by_mean = patient_uncertainty(frame, "mean")
    assert by_max.loc[0, "patient"] == "A"
    assert by_mean.loc[0, "patient"] == "B"


def test_unknown_uncertainty_rule_raises():
    frame = pd.DataFrame({"patient": ["A"], "slice_no": [1], "score": [0.5]})
    with pytest.raises(ValueError, match="unknown uncertainty rule"):
        patient_uncertainty(frame, rule="median")


# ---------------------------------------------------------------------------
# pool bookkeeping
# ---------------------------------------------------------------------------

def test_pool_split_is_a_partition():
    patients = {f"P{i:02d}": i % 3 == 0 for i in range(30)}
    labelled, unlabelled = stratified_pool_split(patients, 0.4, seed=0)
    assert set(labelled) | set(unlabelled) == set(patients)
    assert not set(labelled) & set(unlabelled)      # no patient in both pools


def test_pool_split_is_stratified_on_hemorrhage():
    """A round-0 model trained on an all-negative pool would be useless."""
    patients = {f"P{i:02d}": i < 10 for i in range(30)}   # 10 positive, 20 negative
    labelled, _ = stratified_pool_split(patients, 0.5, seed=1)
    positives = sum(1 for p in labelled if patients[p])
    assert positives == 5                                 # half of the positives
    assert len(labelled) == 15


def test_pool_split_is_reproducible():
    patients = {f"P{i:02d}": i % 2 == 0 for i in range(20)}
    first = stratified_pool_split(patients, 0.4, seed=7)
    second = stratified_pool_split(patients, 0.4, seed=7)
    assert first == second


def test_pool_split_changes_with_the_seed():
    patients = {f"P{i:02d}": i % 2 == 0 for i in range(20)}
    assert stratified_pool_split(patients, 0.4, 1) != stratified_pool_split(patients, 0.4, 2)


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def _ranking(order):
    return pd.DataFrame(
        {"patient": order, "uncertainty": np.linspace(1.0, 0.1, len(order))}
    )


def test_uncertainty_selection_takes_the_most_uncertain():
    ranking = _ranking(["C", "A", "B", "D"])      # C most uncertain
    picked = select_patients(["A", "B", "C", "D"], 2, "uncertainty", 0, ranking)
    assert picked == ["A", "C"]                    # sorted output, C and A chosen


def test_uncertainty_selection_ignores_already_labelled_patients():
    """Only the unlabelled pool is a candidate — re-annotating is wasted budget."""
    ranking = _ranking(["LABELLED", "A", "B"])
    picked = select_patients(["A", "B"], 1, "uncertainty", 0, ranking)
    assert picked == ["A"]
    assert "LABELLED" not in picked


def test_uncertainty_selection_requires_a_ranking():
    with pytest.raises(ValueError, match="requires an uncertainty frame"):
        select_patients(["A"], 1, "uncertainty", 0, None)


def test_random_selection_is_reproducible_and_respects_the_budget():
    pool = [f"P{i:02d}" for i in range(20)]
    first = select_patients(pool, 5, "random", seed=3)
    assert first == select_patients(pool, 5, "random", seed=3)
    assert len(first) == 5
    assert set(first) <= set(pool)


def test_random_selection_differs_from_uncertainty_selection():
    """If the control coincided with the strategy the experiment would be vacuous."""
    pool = [f"P{i:02d}" for i in range(20)]
    ranking = _ranking(pool)
    by_uncertainty = select_patients(pool, 5, "uncertainty", 0, ranking)
    by_random = select_patients(pool, 5, "random", 0)
    assert by_uncertainty != by_random


def test_budget_larger_than_the_pool_is_clamped():
    pool = ["A", "B"]
    assert len(select_patients(pool, 10, "random", 0)) == 2
    assert len(select_patients(pool, 10, "uncertainty", 0, _ranking(pool))) == 2


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown selection strategy"):
        select_patients(["A"], 1, "greedy", 0)


# ---------------------------------------------------------------------------
# experimental hygiene — the part that makes results comparable
# ---------------------------------------------------------------------------

def _index():
    rows = []
    for patient, split in [("P1", "train"), ("P2", "train"), ("P3", "validation"), ("P4", "test")]:
        for slice_no in (1, 2):
            rows.append(
                {
                    "patient": patient, "split": split, "window": "brain",
                    "slice_no": slice_no, "path": f"{patient}/brain/{slice_no}.png",
                    "sha256": "x", "hemorrhage": 0,
                }
            )
    return pd.DataFrame(rows)


def test_restricting_train_leaves_validation_and_test_untouched():
    """Every arm must be judged on identical data — the whole experiment
    depends on this."""
    index = _index()
    restricted = restrict_train_patients(index, {"P1"})

    for split in ("validation", "test"):
        before = index[index["split"] == split].reset_index(drop=True)
        after = restricted[restricted["split"] == split].reset_index(drop=True)
        pd.testing.assert_frame_equal(before, after)


def test_restricting_train_drops_unlabelled_patients():
    restricted = restrict_train_patients(_index(), {"P1"})
    train_patients = set(restricted[restricted["split"] == "train"]["patient"])
    assert train_patients == {"P1"}


def test_expanding_the_pool_only_adds_training_rows():
    index = _index()
    small = restrict_train_patients(index, {"P1"})
    large = restrict_train_patients(index, {"P1", "P2"})
    assert len(large) > len(small)
    # every row of the smaller arm is present in the larger one
    merged = small.merge(large, how="left", indicator=True)
    assert (merged["_merge"] == "both").all()


def test_experiment_index_is_written(tmp_path):
    path = write_experiment_index(tmp_path / "arm", _index())
    assert path.is_file()
    assert pd.read_csv(path).shape[0] == 8


# ---------------------------------------------------------------------------
# summary statistics
# ---------------------------------------------------------------------------

def _results(pairs):
    """pairs: list of (seed, arm, auroc)."""
    return [
        {
            "seed": seed, "arm": arm, "test_auroc": auroc,
            "n_train_patients": 20, "n_train_slices": 600,
        }
        for seed, arm, auroc in pairs
    ]


def test_summary_reports_spread_not_just_a_mean():
    """One flattering number would hide the seed-to-seed variation."""
    summary = summarize(
        _results([(0, "uncertainty", 0.80), (1, "uncertainty", 0.70), (2, "uncertainty", 0.90)])
    )
    block = summary["uncertainty"]
    assert block["n_seeds"] == 3
    assert block["mean_test_auroc"] == pytest.approx(0.80)
    assert block["std_test_auroc"] > 0
    assert block["min_test_auroc"] == 0.70 and block["max_test_auroc"] == 0.90


def test_std_is_none_with_a_single_seed():
    summary = summarize(_results([(0, "uncertainty", 0.8)]))
    assert summary["uncertainty"]["std_test_auroc"] is None


def test_comparison_is_paired_by_seed():
    """Pairing matters: between-seed variance is larger than the effect."""
    summary = summarize(
        _results(
            [
                (0, "uncertainty", 0.85), (0, "random", 0.80),   # +0.05
                (1, "uncertainty", 0.70), (1, "random", 0.75),   # -0.05
                (2, "uncertainty", 0.90), (2, "random", 0.80),   # +0.10
            ]
        )
    )
    comparison = summary["uncertainty_vs_random"]
    assert comparison["paired_deltas"] == [0.05, -0.05, 0.10]
    assert comparison["mean_delta"] == pytest.approx(0.0333, abs=1e-3)
    assert (comparison["wins"], comparison["losses"]) == (2, 1)


def test_comparison_is_absent_without_the_control_arm():
    """No random arm means the active-learning claim cannot be evaluated."""
    summary = summarize(_results([(0, "uncertainty", 0.9), (0, "seed", 0.8)]))
    assert "uncertainty_vs_random" not in summary
    assert "uncertainty_vs_seed" in summary


def test_losses_are_reported_not_hidden():
    summary = summarize(
        _results([(0, "uncertainty", 0.70), (0, "random", 0.80)])
    )
    assert summary["uncertainty_vs_random"]["losses"] == 1
    assert summary["uncertainty_vs_random"]["mean_delta"] < 0
