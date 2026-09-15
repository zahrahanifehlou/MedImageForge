"""Training a baseline, and recording the run so it can be trusted later.

Why the run record matters as much as the model:
    A metric without provenance is a rumour. Every run writes a JSON record
    containing the DATASET VERSION (not a folder path), the code version, the
    git commit, the full configuration, the seed, and the metrics. Months
    later "which data trained this?" is answerable by reading one file — and
    checkable, because the release itself is immutable and verifiable.

Discipline enforced here:
    - The TEST split is touched exactly once, at the very end, using the
      model and threshold selected on VALIDATION. Selecting anything on test
      turns the test score into a training score.
    - Class imbalance is handled with a weighted loss (pos_weight), not by
      throwing away negatives.
    - Every metric is reported next to the trivial baseline, so "88.8%
      accuracy" can never be mistaken for success.
"""

from __future__ import annotations

import json
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from medimageforge import __version__
from medimageforge.data import build_datasets, build_loaders
from medimageforge.logging_utils import get_logger
from medimageforge.metrics import (
    best_threshold,
    bootstrap_auroc_ci,
    classification_metrics,
    trivial_baseline,
)
from medimageforge.model import SliceCNN

log = get_logger(__name__)


@dataclass
class TrainingConfig:
    image_size: int = 128
    batch_size: int = 32
    epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    seed: int = 20260915
    window: str = "brain"


@dataclass
class RunRecord:
    """Everything needed to interpret — and reproduce — one training run."""

    run_id: str
    created_at: str
    code_version: str
    git_commit: str | None
    dataset_version: str
    dataset_index_sha256: str
    config: dict
    environment: dict
    model: dict
    split_sizes: dict = field(default_factory=dict)
    baseline: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    selected_epoch: int | None = None
    selected_threshold: float | None = None
    metrics: dict = field(default_factory=dict)
    duration_seconds: float | None = None


def set_seeds(seed: int) -> None:
    """Make a run as reproducible as CPU PyTorch allows.

    Step 9's lesson applies here too: determinism has to be arranged
    deliberately. Python, NumPy and torch each carry their own RNG.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _git_commit() -> str | None:
    """The code version as a commit hash — a version number is not enough."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        commit = result.stdout.strip()
        return commit or None
    except (OSError, subprocess.SubprocessError):
        return None


@torch.no_grad()
def evaluate(model: nn.Module, loader) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, predicted probabilities) for a whole split."""
    model.eval()
    labels, scores = [], []
    for images, targets in loader:
        logits = model(images.float())
        scores.append(torch.sigmoid(logits).numpy())
        labels.append(targets.numpy())
    return np.concatenate(labels), np.concatenate(scores)


def train_one_epoch(model, loader, optimizer, criterion) -> float:
    model.train()
    total, seen = 0.0, 0
    for images, targets in loader:
        optimizer.zero_grad()
        logits = model(images.float())
        loss = criterion(logits, targets.float())
        loss.backward()
        optimizer.step()
        # .item() detaches first; float(loss) on a grad-tracking tensor warns
        # and would keep the graph alive.
        total += loss.item() * len(targets)
        seen += len(targets)
    return total / max(seen, 1)


def run_training(
    release_dir: Path,
    db_path: Path,
    curated_dir: Path,
    runs_dir: Path,
    config: TrainingConfig,
) -> RunRecord:
    started = time.time()
    set_seeds(config.seed)

    datasets, normalization = build_datasets(
        release_dir,
        db_path,
        curated_dir,
        image_size=config.image_size,
        window=config.window,
        seed=config.seed,
    )
    loaders = build_loaders(datasets, batch_size=config.batch_size)

    train_labels = datasets["train"].labels
    n_pos = float(train_labels.sum())
    n_neg = float(len(train_labels) - n_pos)
    # Weight the positive class by the negative/positive ratio so the
    # minority class is not simply ignored by the loss.
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)])
    log.info(
        "train %d slices (%d positive, pos_weight %.2f)", len(train_labels), int(n_pos),
        float(pos_weight),
    )

    model = SliceCNN(dropout=config.dropout)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    index_bytes = (release_dir / "index.csv").read_bytes()
    import hashlib

    record = RunRecord(
        run_id=datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ"),
        created_at=datetime.now(timezone.utc).isoformat(),
        code_version=__version__,
        git_commit=_git_commit(),
        dataset_version=release_dir.name,
        dataset_index_sha256=hashlib.sha256(index_bytes).hexdigest(),
        config=asdict(config),
        environment={
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "normalization_mean": normalization.mean,
            "normalization_std": normalization.std,
        },
        model={"architecture": "SliceCNN", "parameters": model.count_parameters()},
        split_sizes={k: len(v) for k, v in datasets.items()},
        baseline={
            "test": trivial_baseline(datasets["test"].labels),
            "validation": trivial_baseline(datasets["validation"].labels),
        },
    )

    best_state, best_auroc, best_epoch = None, -1.0, -1
    for epoch in range(1, config.epochs + 1):
        loss = train_one_epoch(model, loaders["train"], optimizer, criterion)
        val_labels, val_scores = evaluate(model, loaders["validation"])
        val_metrics = classification_metrics(val_labels, val_scores)
        record.history.append(
            {
                "epoch": epoch,
                "train_loss": round(loss, 5),
                "val_auroc": round(val_metrics["auroc"], 4),
                "val_f1": round(val_metrics["f1"], 4),
                "val_recall": round(val_metrics["recall"], 4),
            }
        )
        log.info(
            "epoch %02d  loss %.4f  val AUROC %.4f  val F1 %.4f",
            epoch, loss, val_metrics["auroc"], val_metrics["f1"],
        )
        # Model selection on VALIDATION AUROC — threshold-free, so the choice
        # of model is not entangled with the choice of threshold.
        if val_metrics["auroc"] > best_auroc:
            best_auroc, best_epoch = val_metrics["auroc"], epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    record.selected_epoch = best_epoch

    # Threshold is also chosen on validation, never on test.
    val_labels, val_scores = evaluate(model, loaders["validation"])
    threshold, _ = best_threshold(val_labels, val_scores)
    record.selected_threshold = threshold

    # --- the test split is used exactly once, here ------------------------
    record.metrics = {
        "validation": classification_metrics(val_labels, val_scores, threshold),
    }
    test_labels, test_scores = evaluate(model, loaders["test"])
    record.metrics["test"] = classification_metrics(test_labels, test_scores, threshold)

    # Uncertainty, resampling PATIENTS — with 12 test patients the interval is
    # wide, and reporting a point estimate alone would be misleading.
    record.metrics["test"]["auroc_ci"] = bootstrap_auroc_ci(
        test_labels, test_scores, np.array(datasets["test"].patients), seed=config.seed
    )
    record.metrics["validation"]["auroc_ci"] = bootstrap_auroc_ci(
        val_labels, val_scores, np.array(datasets["validation"].patients), seed=config.seed
    )
    record.duration_seconds = round(time.time() - started, 1)

    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / record.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
    torch.save(
        {"state_dict": model.state_dict(), "config": asdict(config), "run_id": record.run_id},
        run_dir / "model.pt",
    )
    np.savez(
        run_dir / "predictions.npz",
        test_labels=test_labels,
        test_scores=test_scores,
        test_patients=np.array(datasets["test"].patients),
        val_labels=val_labels,
        val_scores=val_scores,
        val_patients=np.array(datasets["validation"].patients),
    )
    log.info("run record written to %s", run_dir)
    return record
