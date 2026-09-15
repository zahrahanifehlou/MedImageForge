"""Tests for the Step 10 metrics.

AUROC is hand-implemented here, so it is pinned against cases with known
answers — especially ties, which are the part that breaks silently.
"""

import numpy as np
import pytest

from medimageforge.metrics import (
    auroc,
    best_threshold,
    bootstrap_auroc_ci,
    classification_metrics,
    confusion,
    trivial_baseline,
)


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------

def test_perfect_separation_is_one():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_inverted_ranking_is_zero():
    assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)


def test_constant_scores_give_exactly_half():
    """The tie-handling case. Without average ranks this returns 0.0 or 1.0.

    A model that outputs one constant value carries no information, so the
    only defensible answer is 0.5 — and the trivial baseline depends on it.
    """
    assert auroc([0, 1, 0, 1], [0.7, 0.7, 0.7, 0.7]) == pytest.approx(0.5)


def test_partial_ties_are_averaged():
    # positives at 0.5 and 0.9; negatives at 0.1 and 0.5 -> one tie pair
    # pairs: (0.5 vs 0.1)=1, (0.5 vs 0.5)=0.5, (0.9 vs 0.1)=1, (0.9 vs 0.5)=1
    assert auroc([0, 0, 1, 1], [0.1, 0.5, 0.5, 0.9]) == pytest.approx(3.5 / 4)


def test_auroc_is_half_when_a_class_is_missing():
    assert auroc([1, 1, 1], [0.2, 0.5, 0.9]) == 0.5
    assert auroc([0, 0, 0], [0.2, 0.5, 0.9]) == 0.5


def test_auroc_is_invariant_to_monotone_rescaling():
    labels = [0, 1, 0, 1, 1, 0]
    scores = np.array([0.1, 0.8, 0.3, 0.6, 0.9, 0.2])
    assert auroc(labels, scores) == pytest.approx(auroc(labels, scores * 10 + 5))


# ---------------------------------------------------------------------------
# confusion and derived metrics
# ---------------------------------------------------------------------------

def test_confusion_counts():
    assert confusion([1, 1, 0, 0], [1, 0, 1, 0]) == {"tp": 1, "fn": 1, "fp": 1, "tn": 1}


def test_classification_metrics_are_consistent():
    labels = [1, 1, 1, 0, 0, 0, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1]
    m = classification_metrics(labels, scores, threshold=0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 4)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(4 / 5)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 4 / 5) / 2)
    assert m["accuracy"] == pytest.approx(6 / 8)


def test_metrics_do_not_divide_by_zero_when_nothing_is_predicted_positive():
    m = classification_metrics([1, 0, 0], [0.1, 0.1, 0.1], threshold=0.5)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


# ---------------------------------------------------------------------------
# the trivial baseline — the point of the whole module
# ---------------------------------------------------------------------------

def test_trivial_baseline_shows_accuracy_is_misleading():
    """11.2% prevalence, like the real v1.0 test split."""
    labels = np.array([1] * 40 + [0] * 318)
    baseline = trivial_baseline(labels)
    assert baseline["accuracy"] == pytest.approx(318 / 358)   # ~0.888
    assert baseline["recall"] == 0.0                          # finds nothing
    assert baseline["f1"] == 0.0
    assert baseline["auroc"] == pytest.approx(0.5)            # no information
    assert baseline["balanced_accuracy"] == pytest.approx(0.5)


def test_baseline_predicts_the_majority_class():
    assert trivial_baseline([1, 1, 1, 0])["strategy"] == "always-predict-majority-class"


# ---------------------------------------------------------------------------
# threshold selection
# ---------------------------------------------------------------------------

def test_best_threshold_finds_a_separating_cut():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.6, 0.7]
    threshold, f1 = best_threshold(labels, scores)
    assert f1 == pytest.approx(1.0)
    assert 0.2 < threshold <= 0.6


def test_best_threshold_can_be_below_one_half():
    """Why tuning is needed at all: a model trained on 13% positives rarely
    exceeds 0.5, so a fixed 0.5 cut can yield F1 = 0."""
    labels = [0, 0, 0, 1, 1]
    scores = [0.01, 0.02, 0.03, 0.2, 0.3]
    assert classification_metrics(labels, scores, 0.5)["f1"] == 0.0
    threshold, f1 = best_threshold(labels, scores)
    assert threshold <= 0.2 and f1 == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# bootstrap — the central lesson of Step 10
# ---------------------------------------------------------------------------

def test_patient_bootstrap_is_wider_than_slice_bootstrap():
    """Slices within a patient are correlated, so slice resampling
    understates uncertainty. Measured on the real test split:
    slices [0.689, 0.835] vs patients [0.644, 0.929].
    """
    rng = np.random.default_rng(0)
    labels, scores, patients = [], [], []
    for patient in range(12):
        positive = patient % 3 == 0
        # A per-patient offset is what makes slices within a patient
        # correlated — the whole reason the resampling unit matters. Classes
        # also overlap, so there is real variance to measure (perfectly
        # separated scores would give both methods a zero-width interval).
        patient_effect = rng.normal(0, 0.25)
        for _ in range(30):                      # 30 correlated slices each
            labels.append(int(positive))
            base = 0.55 if positive else 0.45
            scores.append(base + patient_effect + rng.normal(0, 0.05))
            patients.append(f"PAT-{patient:03d}")

    labels = np.array(labels)
    scores = np.array(scores)
    patients = np.array(patients)

    by_slice = bootstrap_auroc_ci(labels, scores, None, n_resamples=400, seed=1)
    by_patient = bootstrap_auroc_ci(labels, scores, patients, n_resamples=400, seed=1)

    slice_width = by_slice["ci_high"] - by_slice["ci_low"]
    patient_width = by_patient["ci_high"] - by_patient["ci_low"]
    # Not merely wider — dramatically so (measured ~5x on this fixture).
    assert patient_width > 2 * slice_width
    assert by_patient["n_units"] == 12
    assert by_patient["resampling_unit"] == "patient"
    assert by_slice["n_units"] == len(labels)


def test_bootstrap_interval_contains_the_point_estimate():
    rng = np.random.default_rng(3)
    labels = np.array([0, 1] * 60)
    scores = rng.random(120) * 0.5 + labels * 0.3
    result = bootstrap_auroc_ci(labels, scores, n_resamples=300, seed=2)
    assert result["ci_low"] <= result["auroc"] <= result["ci_high"]


def test_bootstrap_is_deterministic_given_a_seed():
    rng = np.random.default_rng(4)
    labels = np.array([0, 1] * 40)
    scores = rng.random(80)
    a = bootstrap_auroc_ci(labels, scores, n_resamples=200, seed=7)
    b = bootstrap_auroc_ci(labels, scores, n_resamples=200, seed=7)
    assert a == b


def test_bootstrap_handles_single_class_resamples():
    """With few patients a resample can contain only one class; AUROC is
    undefined there and those draws must be skipped, not crash."""
    labels = np.array([1] * 5 + [0] * 5)
    scores = np.linspace(0, 1, 10)
    groups = np.array(["a"] * 5 + ["b"] * 5)
    result = bootstrap_auroc_ci(labels, scores, groups, n_resamples=50, seed=0)
    assert result["n_resamples"] < 50          # some draws were dropped
    assert 0.0 <= result["ci_low"] <= result["ci_high"] <= 1.0
