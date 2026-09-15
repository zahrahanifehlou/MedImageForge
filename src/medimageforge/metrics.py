"""Classification metrics, chosen for an imbalanced problem.

Why this module exists (the central lesson of Step 10):
    On the v1.0 test split, 11.2% of brain slices show a hemorrhage. So a
    model that ALWAYS answers "no hemorrhage" scores:

        accuracy  0.888   <- looks excellent
        recall    0.000   <- finds nothing
        F1        0.000
        AUROC     0.500   <- no better than a coin flip

    Accuracy is therefore not merely a weak metric here, it is actively
    misleading. `trivial_baseline` computes exactly this so every report has
    the number to beat printed next to it.

AUROC is implemented here rather than pulled from scikit-learn because the
project has no sklearn dependency, and because the tie handling is the part
people get wrong: with ties you must use AVERAGE ranks, otherwise a model
that outputs one constant score scores 0.0 or 1.0 instead of 0.5.
"""

from __future__ import annotations

import numpy as np


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, via the Mann-Whitney U statistic.

    AUROC = P(score of a random positive > score of a random negative),
    with ties counted as half. Threshold-free, so it does not depend on
    where we set the decision boundary, and unaffected by class prevalence.

    Returns 0.5 when either class is absent — undefined, but 0.5 is the
    honest "no information" answer and keeps callers simple.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Average ranks for tied scores: this is what makes a constant-score
    # model land on exactly 0.5 instead of 0.0 or 1.0.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0     # ranks are 1-based
        ranks[order[i : j + 1]] = average_rank
        i = j + 1

    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def confusion(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    labels = np.asarray(labels).astype(int)
    predictions = np.asarray(predictions).astype(int)
    return {
        "tp": int(((predictions == 1) & (labels == 1)).sum()),
        "fp": int(((predictions == 1) & (labels == 0)).sum()),
        "tn": int(((predictions == 0) & (labels == 0)).sum()),
        "fn": int(((predictions == 0) & (labels == 1)).sum()),
    }


def classification_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Everything needed to judge an imbalanced binary classifier."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= threshold).astype(int)
    counts = confusion(labels, predictions)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0          # sensitivity
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        **{k: float(v) for k, v in counts.items()},
        "n": float(len(labels)),
        "prevalence": float(labels.mean()) if len(labels) else 0.0,
        "accuracy": (tp + tn) / len(labels) if len(labels) else 0.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        # The honest accuracy-like number under imbalance: the mean of
        # per-class recalls, so ignoring the minority class cannot hide.
        "balanced_accuracy": (recall + specificity) / 2,
        "f1": f1,
        "auroc": auroc(labels, scores),
        "threshold": float(threshold),
    }


def trivial_baseline(labels: np.ndarray) -> dict[str, float]:
    """The majority-class predictor — what any real model must beat.

    Scores are constant, so AUROC is exactly 0.5 (thanks to tie handling).
    """
    labels = np.asarray(labels).astype(int)
    majority = 1.0 if labels.mean() > 0.5 else 0.0
    scores = np.full(len(labels), majority, dtype=np.float64)
    metrics = classification_metrics(labels, scores, threshold=0.5)
    metrics["strategy"] = "always-predict-majority-class"
    return metrics


def bootstrap_auroc_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray | None = None,
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Confidence interval for AUROC by resampling — PATIENTS, not slices.

    Why `groups` matters more than the number of resamples
    ------------------------------------------------------
    Slices from one patient are not independent observations: they are ~30
    images of the same head. Resampling slices therefore pretends we have 358
    independent data points when we effectively have 12. Measured on the v1.0
    test split, the same AUROC of 0.768 yields:

        resampling slices   95% CI [0.689, 0.835]   <- falsely narrow
        resampling patients 95% CI [0.644, 0.929]   <- honest

    Under-reporting uncertainty is the same mistake as splitting by slice,
    one step later in the pipeline. Pass `groups` (patient ids) whenever you
    have them.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)

    if groups is None:
        units = np.arange(len(labels))
        index_of = {u: np.array([u]) for u in units}
    else:
        groups = np.asarray(groups)
        units = np.unique(groups)
        index_of = {u: np.where(groups == u)[0] for u in units}

    values = []
    for _ in range(n_resamples):
        chosen = rng.choice(units, size=len(units), replace=True)
        idx = np.concatenate([index_of[u] for u in chosen])
        if len(np.unique(labels[idx])) < 2:
            continue                        # AUROC undefined for one class
        values.append(auroc(labels[idx], scores[idx]))

    # Every return path carries the same keys — a caller should never have to
    # branch on whether the degenerate case was hit.
    result = {
        "auroc": round(auroc(labels, scores), 4),
        "n_units": int(len(units)),
        "resampling_unit": "patient" if groups is not None else "slice",
        "n_resamples": len(values),
    }
    if not values:
        # Happens when every resample contains a single class, e.g. a split
        # with one patient. An interval of [0, 1] is the honest answer:
        # this data supports no conclusion at all.
        return {**result, "ci_low": 0.0, "ci_high": 1.0, "undefined": True}
    return {
        **result,
        "ci_low": round(float(np.percentile(values, 100 * alpha / 2)), 4),
        "ci_high": round(float(np.percentile(values, 100 * (1 - alpha / 2))), 4),
    }


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Threshold maximizing F1, chosen on VALIDATION only.

    Why it must be tuned at all: 0.5 is arbitrary for an imbalanced problem —
    a well-calibrated model trained on 13% positives rarely exceeds 0.5.
    Why validation only: picking it on test would report a number nobody
    could reproduce on new data.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    candidates = np.unique(np.concatenate([scores, [0.0, 1.0]]))
    best, best_f1 = 0.5, -1.0
    for threshold in candidates:
        f1 = classification_metrics(labels, scores, float(threshold))["f1"]
        if f1 > best_f1:
            best, best_f1 = float(threshold), f1
    return best, best_f1
