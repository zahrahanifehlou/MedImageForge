"""Tests for the Step 3 image inspection utilities.

Synthetic arrays for the pure functions (overlay, mask recovery, layout);
the real dataset only for load_gray — the one thing that must match reality.
"""

import numpy as np

from medimageforge.config import data_path, load_config
from medimageforge.imaging import (
    load_gray,
    mask_from_jpeg,
    overlay_mask,
    pixel_stats,
    side_by_side,
)


def test_load_gray_real_slice():
    config = load_config()
    arr = load_gray(data_path(config, "raw_dir") / "049" / "brain" / "14.jpg")
    assert arr.shape == (650, 650)
    assert arr.dtype == np.uint8


def test_mask_threshold_recovers_binary_region():
    # JPEG-style mask: mostly 0/255 with ringing artifacts near edges.
    noisy = np.array([[0, 12, 200], [0, 255, 90], [0, 3, 255]], dtype=np.uint8)
    mask = mask_from_jpeg(noisy, threshold=128)
    assert mask.dtype == bool
    assert mask.tolist() == [
        [False, False, True],
        [False, True, False],
        [False, False, True],
    ]


def test_overlay_colors_only_masked_pixels():
    image = np.full((4, 4), 100, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    out = overlay_mask(image, mask, color=(255, 0, 0), alpha=1.0)
    assert out.shape == (4, 4, 3)
    assert tuple(out[1, 1]) == (255, 0, 0)   # masked pixel turned red
    assert tuple(out[0, 0]) == (100, 100, 100)  # untouched pixel stays gray


def test_side_by_side_layout():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.zeros((10, 10, 3), dtype=np.uint8)
    out = side_by_side(a, b)
    # 10 + 2px separator + 10 wide, always RGB
    assert out.shape == (10, 22, 3)


def test_pixel_stats():
    arr = np.array([[0, 255], [0, 255]], dtype=np.uint8)
    stats = pixel_stats(arr)
    assert stats["min"] == 0 and stats["max"] == 255
    assert stats["mean"] == 127.5
