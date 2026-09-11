# Step 3 — Image inspection

## What we built

`src/medimageforge/imaging.py` (pure, testable functions) plus an `inspect`
subcommand:

```bash
python -m medimageforge inspect 049          # auto-picks first masked slice
python -m medimageforge inspect 049 --slice 14
```

It writes `artifacts/inspect/<patient>_<slice>.png`: three panels —
bone window | brain window | brain window with the `_HGE_Seg` mask overlaid
in red — and prints pixel statistics for each input.

## Why it exists

Step 2 proved the files *exist* and are byte-correct. Step 3 proves we
understand what the pixels *mean* — and gives the visual confirmation that a
mask aligns with an actual hemorrhage. If the overlay didn't sit on a bright
spot in the brain window, we'd know the label↔pixel mapping was wrong before
any pipeline depended on it.

## How it works

```text
inspect 049 14
   ├─ load_gray()         brain/14.jpg, bone/14.jpg → uint8 (650,650) arrays
   ├─ mask_from_jpeg()    14_HGE_Seg.jpg → bool mask (threshold >128)
   ├─ overlay_mask()      blend red into an RGB copy where mask is True
   └─ side_by_side()      concat [bone | brain | overlay] → artifacts/...png
```

## What the three panels teach

- **CT windowing.** Same slice, two images. Bone window spends its contrast
  on dense bone: skull crisp, brain a featureless wash. Brain window spends
  it on soft tissue: hemorrhage (fresh blood is *hyperdense* → bright)
  becomes visible. One scan can never show both well — that is why the
  dataset ships two windows.
- **Segmentation masks.** A mask is a same-size image, white where the
  hemorrhage is. Saved as JPEG it is not perfectly binary (compression
  ringing), so `mask_from_jpeg` thresholds at the midpoint.
- **What the model will see.** uint8 grayscale 650×650, 0–255. On slice
  049/14 the mask is only 687 px (0.2% of the image) — a reminder that
  hemorrhages can be tiny, which will matter for both preprocessing and
  metrics in Phase 4.

## Concepts learned

- **Pixels are data with geometry** — stats (mean/std/range) describe the
  array; only the visual check ties them to anatomy.
- **Artifacts vs raw:** composites go in `artifacts/` (regenerable output),
  never written next to the raw data. Raw stays immutable.
- **Everything testable:** overlay, masking, layout, stats are pure functions
  on numpy arrays — no file I/O needed to test them.

## Verified

- `artifacts/inspect/049_14.png` shows the red overlay on the hyperdense
  (bright) region in the brain window — mask aligns with the hemorrhage
- `pytest` → 15 passed
