"""Tests for the Step 10 data loading, model, and training run.

Focus: the properties that make a run trustworthy — release-driven loading,
pseudonym resolution, no normalization leakage, and a complete run record.
"""

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from medimageforge.data import (
    Normalization,
    SliceDataset,
    build_datasets,
    build_loaders,
    load_pseudonym_map,
    resolve_paths,
)
from medimageforge.manifest import connect
from medimageforge.model import SliceCNN
from medimageforge.privacy import PATIENTS_SCHEMA

torch = pytest.importorskip("torch")


@pytest.fixture
def release(tmp_path):
    """A tiny release: 6 patients x (4 brain + 4 bone) slices, plus manifest."""
    curated = tmp_path / "curated"
    release_dir = tmp_path / "datasets" / "v1.0"
    release_dir.mkdir(parents=True)
    db = tmp_path / "manifest.db"

    rng = np.random.default_rng(0)
    rows = []
    with connect(db) as conn:
        conn.executescript(PATIENTS_SCHEMA)
        # 8 patients so validation and test each hold TWO patients, one of
        # each class — otherwise AUROC is undefined on those splits.
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
            for window in ("brain", "bone"):
                for slice_no in range(1, 5):
                    # positive slices are brighter, so a model can learn something
                    base = 200 if positive else 60
                    array = np.clip(
                        rng.normal(base, 10, (32, 32)), 0, 255
                    ).astype(np.uint8)
                    path = curated / real_id / window / f"{slice_no}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(array).save(path)
                    rows.append(
                        {
                            "patient": pseudonym,
                            "split": split,
                            "window": window,
                            "slice_no": slice_no,
                            "path": f"{pseudonym}/{window}/{slice_no}.png",
                            "sha256": "x",
                            "hemorrhage": int(positive),
                        }
                    )
        conn.commit()
    pd.DataFrame(rows).to_csv(release_dir / "index.csv", index=False)
    return release_dir, db, curated


# ---------------------------------------------------------------------------
# pseudonym resolution — the Step 9 design in action
# ---------------------------------------------------------------------------

def test_pseudonym_map_is_read_from_the_manifest(release):
    _, db, _ = release
    mapping = load_pseudonym_map(db)
    assert mapping["PAT-000000000001"] == "001"


def test_missing_pseudonym_map_raises_a_helpful_error(tmp_path):
    with pytest.raises(RuntimeError, match="privacy"):
        load_pseudonym_map(tmp_path / "empty.db")


def test_release_paths_resolve_to_real_curated_files(release):
    release_dir, db, curated = release
    index = pd.read_csv(release_dir / "index.csv")
    paths = resolve_paths(index, load_pseudonym_map(db), curated)
    assert all(p.is_file() for p in paths)
    # the real file lives under the REAL id, not the pseudonym
    assert "001/brain" in str(paths[0]) or "001/bone" in str(paths[0])


def test_unknown_pseudonym_is_an_error(release):
    _, db, curated = release
    index = pd.DataFrame({"path": ["PAT-doesnotexist/brain/1.png"]})
    with pytest.raises(KeyError, match="resolve"):
        resolve_paths(index, load_pseudonym_map(db), curated)


# ---------------------------------------------------------------------------
# dataset construction
# ---------------------------------------------------------------------------

def test_only_the_configured_window_is_loaded(release):
    release_dir, db, curated = release
    datasets, _ = build_datasets(release_dir, db, curated, image_size=32, window="brain")
    # 4 train patients x 4 brain slices; 2 patients each in val/test
    assert len(datasets["train"]) == 16
    assert len(datasets["validation"]) == 8
    assert len(datasets["test"]) == 8


def test_normalization_is_fitted_on_train_only(release):
    """Fitting on all splits would leak test information into training."""
    release_dir, db, curated = release
    datasets, normalization = build_datasets(
        release_dir, db, curated, image_size=32, window="brain"
    )
    train_pixels = datasets["train"].raw_pixels()
    assert normalization.mean == pytest.approx(float(train_pixels.mean()))
    assert normalization.std == pytest.approx(float(train_pixels.std()))

    all_pixels = np.concatenate(
        [datasets[s].raw_pixels().ravel() for s in ("train", "validation", "test")]
    )
    # the train-only mean must differ from the all-splits mean, otherwise this
    # test would pass even if the code leaked
    assert normalization.mean != pytest.approx(float(all_pixels.mean()))


def test_all_splits_share_the_train_normalization(release):
    release_dir, db, curated = release
    datasets, normalization = build_datasets(
        release_dir, db, curated, image_size=32, window="brain"
    )
    for dataset in datasets.values():
        assert dataset.normalization is normalization


def test_items_have_the_expected_shape_and_dtype(release):
    release_dir, db, curated = release
    datasets, _ = build_datasets(release_dir, db, curated, image_size=32, window="brain")
    image, label = datasets["train"][0]
    assert image.shape == (1, 32, 32)          # channel-first, single channel
    assert image.dtype == np.float32
    assert label in (0.0, 1.0)


def test_augmentation_is_train_only(release):
    release_dir, db, curated = release
    datasets, _ = build_datasets(release_dir, db, curated, image_size=32, window="brain")
    assert datasets["train"].augment is True
    assert datasets["validation"].augment is False
    assert datasets["test"].augment is False


def test_horizontal_flip_preserves_the_label():
    """Augmentation must not change the answer it is training towards."""
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    dataset = SliceDataset.__new__(SliceDataset)
    dataset.images = image[None, :, :]
    dataset.labels = np.array([1.0], dtype=np.float32)
    dataset.image_size = 4
    dataset.normalization = None
    dataset.augment = True
    dataset._rng = np.random.default_rng(0)
    dataset.paths = [None]
    _, label = dataset[0]
    assert label == 1.0


def test_normalization_apply_standardizes():
    norm = Normalization(mean=0.5, std=0.25)
    out = norm.apply(np.array([0.5, 0.75], dtype=np.float32))
    assert out[0] == pytest.approx(0.0, abs=1e-5)
    assert out[1] == pytest.approx(1.0, abs=1e-4)


def test_only_train_loader_shuffles(release):
    release_dir, db, curated = release
    datasets, _ = build_datasets(release_dir, db, curated, image_size=32, window="brain")
    loaders = build_loaders(datasets, batch_size=4)
    from torch.utils.data import RandomSampler, SequentialSampler

    assert isinstance(loaders["train"].sampler, RandomSampler)
    assert isinstance(loaders["test"].sampler, SequentialSampler)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def test_model_returns_one_logit_per_image():
    model = SliceCNN()
    out = model(torch.zeros(5, 1, 64, 64))
    assert out.shape == (5,)


def test_model_is_resolution_independent():
    """Global average pooling means input size is not baked in."""
    model = SliceCNN()
    assert model(torch.zeros(2, 1, 32, 32)).shape == (2,)
    assert model(torch.zeros(2, 1, 128, 128)).shape == (2,)


def test_model_outputs_are_logits_not_probabilities():
    model = SliceCNN()
    with torch.no_grad():
        out = model(torch.randn(32, 1, 32, 32))
    # logits are unbounded; probabilities would sit inside [0, 1]
    assert out.min() < 0.0 or out.max() > 1.0


def test_parameter_count_is_small():
    assert 10_000 < SliceCNN().count_parameters() < 500_000


# ---------------------------------------------------------------------------
# an end-to-end run and its record
# ---------------------------------------------------------------------------

@pytest.fixture
def trained(release, tmp_path):
    from medimageforge.train import TrainingConfig, run_training

    release_dir, db, curated = release
    config = TrainingConfig(image_size=32, batch_size=4, epochs=2, seed=7)
    record = run_training(release_dir, db, curated, tmp_path / "runs", config)
    return record, tmp_path / "runs"


def test_run_record_identifies_the_dataset_version_not_a_folder(trained):
    """The reason the record exists: 'trained on v1.0' must be checkable."""
    record, _ = trained
    assert record.dataset_version == "v1.0"
    assert len(record.dataset_index_sha256) == 64


def test_run_record_is_complete(trained):
    record, runs_dir = trained
    assert record.code_version
    assert record.config["seed"] == 7
    assert record.environment["torch"]
    assert record.model["parameters"] > 0
    assert record.split_sizes["train"] > 0
    assert record.selected_epoch in (1, 2)
    assert record.selected_threshold is not None
    assert "test" in record.metrics and "validation" in record.metrics
    assert record.baseline["test"]["auroc"] == pytest.approx(0.5)
    assert record.duration_seconds >= 0


def test_run_record_is_persisted_as_json(trained):
    record, runs_dir = trained
    payload = json.loads((runs_dir / record.run_id / "run.json").read_text())
    assert payload["dataset_version"] == "v1.0"
    assert payload["metrics"]["test"]["auroc"] >= 0.0
    assert (runs_dir / record.run_id / "model.pt").is_file()
    assert (runs_dir / record.run_id / "predictions.npz").is_file()


def test_predictions_are_saved_with_patient_ids(trained):
    """Needed for patient-level bootstrapping and for Step 11's analysis."""
    record, runs_dir = trained
    data = np.load(runs_dir / record.run_id / "predictions.npz", allow_pickle=True)
    assert len(data["test_labels"]) == len(data["test_scores"])
    assert len(data["test_patients"]) == len(data["test_labels"])
    assert all(str(p).startswith("PAT-") for p in data["test_patients"])


def test_auroc_ci_is_recorded_over_patients(trained):
    record, _ = trained
    ci = record.metrics["test"]["auroc_ci"]
    assert ci["resampling_unit"] == "patient"
    assert ci["ci_low"] <= ci["auroc"] <= ci["ci_high"]


def test_training_is_reproducible(release, tmp_path):
    """Same seed, same data, same result — Step 9's lesson applied to training."""
    from medimageforge.train import TrainingConfig, run_training

    release_dir, db, curated = release
    config = TrainingConfig(image_size=32, batch_size=4, epochs=2, seed=11)
    first = run_training(release_dir, db, curated, tmp_path / "a", config)
    second = run_training(release_dir, db, curated, tmp_path / "b", config)
    assert [h["train_loss"] for h in first.history] == [
        h["train_loss"] for h in second.history
    ]
    assert first.metrics["test"]["auroc"] == second.metrics["test"]["auroc"]
