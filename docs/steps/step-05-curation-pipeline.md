# Step 5 — Curation pipeline

```bash
python -m medimageforge curate           # validate + normalize into the curated zone
python -m medimageforge curate --force   # re-curate everything from scratch
```

---

## 1. What we built

`src/medimageforge/curate.py` plus a `curate` subcommand that:

1. reads the file list **from the manifest** (not by re-walking the disk),
2. **validates** every image against explicit rules,
3. writes **normalized, lossless copies** into `artifacts/curated/`,
4. records every decision — with its reason — in a new `curation` table,
5. writes `artifacts/curation_report.json`.

## 2. Why it exists

Raw data is not ML-ready. Before a model ever sees a file we must answer:
is it readable? the expected shape? a duplicate? does it have a label?
A file that fails must be **excluded with a recorded reason** — never
silently dropped, because "why is this slice missing from training?" has to
be answerable months later.

---

## 3. The two design rules that matter more than the code

### Rule 1 — Zone separation

We **read** `data/` and **write** `artifacts/curated/`. Raw is never modified.

```text
data/                          artifacts/curated/
  Patients_CT/049/brain/         049/brain/
    14.jpg          ── read ──►    14.png        (lossless copy)
    14_HGE_Seg.jpg  ── read ──►    14_mask.png   (true binary mask)
  (immutable)                    (regenerable — delete and re-run anytime)
```

Consequence: any curation bug is fixed by deleting the curated zone and
re-running. Raw is always the fallback. A test enforces this
(`test_raw_zone_is_never_modified` hashes the whole raw tree before and after).

### Rule 2 — Reject vs warn

Not every anomaly is a defect.

| Decision | Meaning | Example found in this dataset |
|---|---|---|
| **rejected** | unusable — excluded from the curated zone | (none: the data is clean) |
| **warning** | usable but noteworthy — still curated | `084/brain/36.jpg` has no bone counterpart |

Patient 084's brain slice 36 is a perfectly good image; it is only missing its
bone-window twin. Rejecting it would throw away real data for no reason.
Conflating "anomalous" with "unusable" is how datasets quietly shrink.

---

## 4. The validation rules, and the failure each one catches

| Rule | Rejection reason | Real-world cause |
|---|---|---|
| PIL can decode it (forced `im.load()`) | `unreadable:<Error>` | truncated download, not an image |
| exactly 650×650 | `wrong-dimensions:(w,h)` | a model expecting 650² crashes or mis-crops |
| has information content (`max>0`, `std≥1`) | `blank-or-constant` | all-black frame, failed export |
| content hash is unique | `duplicate-of:<path>` | same image copied twice → train/test leakage |
| slice has a label row | `no-label-row` | unusable for supervised training |
| mask has its slice | `mask-without-slice` | orphan annotation |

`expected_size` and `mask_threshold` live in `configs/default.yaml`, not in
code — same principle as Step 1.

---

## 5. Normalization: JPEG → PNG, and why it is not cosmetic

**JPEG is lossy.** Every re-save degrades the image. The curated zone is PNG
(lossless) so no downstream step ever re-compresses. Verified:

```text
slice: curated PNG pixels == raw JPEG pixels exactly  (np.array_equal → True)
```

**Masks get genuinely fixed.** Step 3 showed `_HGE_Seg.jpg` is not truly
binary — JPEG ringing leaves values like 1, 2, 3… Curation thresholds it once,
centrally, instead of every consumer re-deriving it:

```text
raw     14_HGE_Seg.jpg :  50 distinct pixel values  [0 1 2 3 4 5 6 7 8 9 10 11 ...]
curated 14_mask.png    :   2 distinct pixel values  [0 255]
white pixels           : 687 in both — the region is preserved, only cleaned
```

The trade-off, stated honestly: raw is 114 MB, curated is 333 MB. Lossless
costs disk. We accept that because correctness beats storage at this scale —
and because re-compressing medical images repeatedly is indefensible.

---

## 6. Provenance: the `curation` table

Every decision is recorded, keyed by source path:

| Column | Purpose |
|---|---|
| `decision` | `accepted` / `rejected` |
| `reasons` | why rejected (empty when accepted) |
| `warnings` | usable, but noteworthy |
| `curated_path` | where the output went (NULL if rejected) |
| `source_sha256` | **which version** of the raw file this decision was about |
| `curated_sha256` | fingerprint of what we produced |
| `curated_at` | when |

`source_sha256` is the provenance link: if a raw file ever changes, its old
curation decision no longer applies — and that is exactly how the idempotency
check knows to redo the work. `files.status` is also updated to
`curated` / `rejected`, so the manifest stays the single place to ask
"what state is this file in?".

---

## 7. Idempotency, and a subtle bug it exposed

Re-running skips files whose `source_sha256` is unchanged:

```text
run 1:  Accepted 5319 | Rejected 0 | Skipped 0
run 2:  Accepted    0 | Rejected 0 | Skipped 5319
```

**The bug this created.** The first version of the JSON report described only
the *last run*. After an idempotent re-run it read:

```json
{"accepted": 0, "rejected": 0, "skipped": 5319, "warnings": []}
```

Zero warnings — so a reviewer would conclude the dataset is spotless, when
`084/brain/36` is still unpaired. **A report of a run is not a report of the
dataset.** Fixed by adding `curation_state()`, which reads all decisions back
out of the manifest. The report now carries both views:

```json
{
  "last_run":      {"accepted": 0, "rejected": 0, "skipped": 5319},
  "dataset_state": {"accepted": 5319, "rejected": 0,
                    "warnings": [{"rel_path": "Patients_CT/084/brain/36.jpg",
                                  "warnings": ["missing-bone-counterpart"]}]}
}
```

`last_run` answers *"what did I just do?"*. `dataset_state` answers
*"what is in the curated zone?"*. Both matter; they are not the same question.

---

## 8. Results on the real dataset

| Metric | Value |
|---|---|
| Accepted | 5319 (all images) |
| Rejected | 0 |
| Warnings | 1 — `084/brain/36.jpg` missing bone counterpart |
| `files.status` | 5319 `curated`, 7 `raw` (the metadata files aren't images) |
| Curated zone | 333 MB of PNG, zero JPEG files |

**Zero rejections is an honest result, not a broken gate.** Step 2/3
measurements already showed all 5,319 images are 650×650 mode `L`, readable,
non-blank, hash-unique, and fully labelled. So the gates are proven by
synthetic tests instead: `tests/test_curate.py` builds a raw zone containing
one of every failure mode and asserts each rejection fires.

A real bug surfaced this way: the rejection `INSERT` was missing a binding and
would have crashed on the first genuinely bad file. The clean dataset never
triggered it. **An untested gate is an assumption, not a control.**

---

## 9. Concepts learned

- **Zone separation** — raw immutable, derived data regenerable.
- **Reject vs warn** — usable-but-odd data must survive curation.
- **Lossless normalization** — fix a data defect (non-binary masks) once,
  centrally, rather than in every consumer.
- **Provenance by content hash** — a decision is about a *specific version*
  of a file, which is what makes idempotency correct rather than lucky.
- **Run log ≠ state snapshot** — the distinction that made our first report
  misleading.
- **Test the paths your data never takes** — clean data hides broken gates.

## 10. Verified

- `curate` → 5319 accepted, 0 rejected, 1 warning; re-run skips all 5319
- curated masks are exactly `{0, 255}`; curated slices bit-identical to source
- no `.jpg` anywhere in the curated zone; raw tree byte-for-byte unchanged
- `pytest` → 35 passed
