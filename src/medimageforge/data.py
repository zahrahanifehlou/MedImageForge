"""PyTorch data loading from a dataset RELEASE (not from a folder).

Why this module exists:
    This is the seam between the data platform and the model. The important
    detail is what it reads: `datasets/v1.0/index.csv`, a versioned, immutable
    release — never a directory scan. That is what makes an experiment record
    meaningful later: "trained on v1.0" is checkable; "trained on curated/"
    is not.

Three things worth understanding here:

    1. Pseudonym resolution. The release references images by pseudonymous
       path (`PAT-98c32.../brain/14.png`) because a release leaves the
       controlled zone. The real file lives at `curated/049/brain/14.png`, so
       the loader must resolve pseudonym -> real ID through the manifest's
       `patients` table. Without the manifest, the release is unusable — the
       re-identification control from Step 6 working as designed.

    2. Normalization statistics come from the TRAIN split only. Computing a
       mean/std over the whole dataset would let test-set information leak
       into training. It is a small leak, but it is still a leak, and it is
       the kind nobody notices.

    3. Brain window only. Hemorrhage is visible in the brain window (Step 3);
       the bone window exists for fractures. Feeding both to a
       hemorrhage classifier would double the data with half of it
       near-uninformative for this task.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from medimageforge.logging_utils import get_logger
from medimageforge.privacy import read_pseudonym_map

log = get_logger(__name__)


def load_pseudonym_map(db_path: Path) -> dict[str, str]:
    """pseudonym -> real patient id, read from the manifest's patients table.

    Reads the persisted table rather than recomputing HMACs: path resolution
    needs the mapping, not the secret salt. The fewer components that touch
    the key, the better — and a reader must never rewrite what it reads.
    """
    forward = read_pseudonym_map(db_path)
    if not forward:
        raise RuntimeError(
            "No pseudonym map in the manifest — run "
            "`python -m medimageforge privacy` to create it."
        )
    return {pseudonym: real_id for real_id, pseudonym in forward.items()}


def resolve_paths(
    index: pd.DataFrame, pseudonym_map: dict[str, str], curated_dir: Path
) -> list[Path]:
    """Turn pseudonymous release paths into real curated file paths."""
    paths = []
    for rel_path in index["path"]:
        pseudonym, _, remainder = str(rel_path).partition("/")
        real_id = pseudonym_map.get(pseudonym)
        if real_id is None:
            raise KeyError(f"Cannot resolve pseudonym {pseudonym!r} — wrong manifest?")
        paths.append(curated_dir / real_id / remainder)
    return paths


@dataclass
class Normalization:
    """Per-dataset intensity statistics, fitted on the train split alone."""

    mean: float
    std: float

    def apply(self, images: np.ndarray) -> np.ndarray:
        return (images - self.mean) / (self.std + 1e-8)


class SliceDataset(Dataset):
    """Slice-level hemorrhage classification: one brain slice -> 0/1.

    Images are decoded once into memory. At 128x128 the whole brain-window
    set is ~41 MB, so caching removes the file-IO bottleneck that would
    otherwise dominate CPU training.
    """

    def __init__(
        self,
        paths: list[Path],
        labels: np.ndarray,
        patients: list[str],
        slice_numbers: list[int] | None = None,
        image_size: int = 128,
        normalization: Normalization | None = None,
        augment: bool = False,
        seed: int = 0,
    ):
        self.paths = paths
        self.labels = labels.astype(np.float32)
        self.patients = patients
        # Slice numbers are carried alongside patients so a prediction can be
        # traced back to a specific image. Without them, error analysis can
        # only say "some slice of this patient was wrong", and predictions
        # cannot be joined to per-subtype labels in the release index.
        self.slice_numbers = (
            list(slice_numbers) if slice_numbers is not None else list(range(len(paths)))
        )
        self.image_size = image_size
        self.normalization = normalization
        self.augment = augment
        self._rng = np.random.default_rng(seed)
        self.images = self._decode_all()

    def _decode_all(self) -> np.ndarray:
        buffer = np.zeros((len(self.paths), self.image_size, self.image_size), dtype=np.uint8)
        for i, path in enumerate(self.paths):
            with Image.open(path) as im:
                resized = im.convert("L").resize(
                    (self.image_size, self.image_size), Image.BILINEAR
                )
                buffer[i] = np.asarray(resized, dtype=np.uint8)
        return buffer

    def raw_pixels(self) -> np.ndarray:
        """Decoded images scaled to [0, 1] — used to fit normalization."""
        return self.images.astype(np.float32) / 255.0

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = self.images[idx].astype(np.float32) / 255.0
        if self.augment and self._rng.random() < 0.5:
            # Horizontal flip only. A head CT is roughly left-right
            # symmetric, and for "is there a hemorrhage anywhere in this
            # slice" the side does not change the answer. Vertical flips or
            # large rotations would produce anatomically impossible images.
            image = image[:, ::-1].copy()
        if self.normalization:
            image = self.normalization.apply(image)
        return image[None, :, :], self.labels[idx]


def build_datasets(
    release_dir: Path,
    db_path: Path,
    curated_dir: Path,
    image_size: int = 128,
    window: str = "brain",
    seed: int = 0,
) -> tuple[dict[str, SliceDataset], Normalization]:
    """Build train/validation/test datasets from a release index."""
    index = pd.read_csv(release_dir / "index.csv")
    index = index[index["window"] == window].reset_index(drop=True)
    pseudonym_map = load_pseudonym_map(db_path)

    datasets: dict[str, SliceDataset] = {}
    for split in ("train", "validation", "test"):
        subset = index[index["split"] == split].reset_index(drop=True)
        datasets[split] = SliceDataset(
            paths=resolve_paths(subset, pseudonym_map, curated_dir),
            labels=subset["hemorrhage"].to_numpy(),
            patients=subset["patient"].tolist(),
            slice_numbers=subset["slice_no"].tolist(),
            image_size=image_size,
            augment=(split == "train"),
            seed=seed,
        )

    # Fit on TRAIN ONLY, then share with the other splits.
    train_pixels = datasets["train"].raw_pixels()
    normalization = Normalization(
        mean=float(train_pixels.mean()), std=float(train_pixels.std())
    )
    for dataset in datasets.values():
        dataset.normalization = normalization
    return datasets, normalization


def build_loaders(
    datasets: dict[str, SliceDataset], batch_size: int = 32
) -> dict[str, DataLoader]:
    """DataLoaders. num_workers=0 keeps the run deterministic and, with the
    images already cached in memory, costs nothing."""
    return {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            drop_last=False,
        )
        for split, dataset in datasets.items()
    }
