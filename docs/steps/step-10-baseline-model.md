# Step 10 — Baseline model

```bash
pip install torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m medimageforge train              # ~70 s on CPU
python -m medimageforge train --epochs 2
```

This opens Phase 4.

---

## 1. What & why

A slice-level classifier (hemorrhage vs no hemorrhage) trained on the **v1.0
release**, plus a run record that makes the result interpretable months later.

The chain, and where each piece came from:

```text
datasets/v1.0/index.csv        ← Step 9 release (immutable, versioned)
      │  resolve PAT-... → real curated path via the manifest (Step 6)
      ▼
SliceDataset  (brain window, 128×128, train-only normalization)
      ▼
DataLoader → SliceCNN (97,521 params) → BCEWithLogitsLoss(pos_weight)
      ▼
metrics vs the trivial baseline  +  patient-level bootstrap CI
      ▼
artifacts/runs/<run-id>/{run.json, model.pt, predictions.npz}
```

**Training reads a release, never a folder.** That is what makes "trained on
v1.0" a checkable claim: the release is immutable and verifiable, so the
record's `dataset_version` + `dataset_index_sha256` pin the exact data.

---

## 2. The metric lesson: accuracy is actively misleading here

11.2% of test brain slices show a hemorrhage. So the do-nothing model scores:

| metric | always-predict-"no hemorrhage" | our model |
|---|---|---|
| **accuracy** | **0.8883** | 0.7989 |
| AUROC | 0.5000 | **0.7681** |
| balanced accuracy | 0.5000 | **0.6355** |
| recall | 0.0000 | **0.4250** |
| F1 | 0.0000 | **0.3208** |

**Our model is far better and its accuracy is 9 points WORSE.** A model that
never predicts a hemorrhage is useless in a stroke workup, yet wins on
accuracy. This is why `trivial_baseline` is computed on every run and printed
next to the model — the number to beat is always visible.

AUROC leads the table because it is threshold-free and prevalence-independent.
It is implemented by hand (no sklearn dependency) with **average ranks for
ties**, which is what makes a constant-score model score exactly 0.5 rather
than 0.0 or 1.0 — the trivial baseline depends on that detail.

---

## 3. The uncertainty lesson: the resampling unit must match the unit of independence

Point estimate: test AUROC **0.7681**. Two bootstrap intervals for that same
number:

| resampling unit | 95% CI | width |
|---|---|---|
| slices (naive) | [0.689, 0.835] | 0.15 |
| **patients (correct)** | **[0.651, 0.936]** | **0.29** |

358 test slices come from **12 patients**. Slices within a patient are ~30
images of one head, so resampling slices pretends we have 358 independent
observations when we effectively have 12 — it understates uncertainty. This is
**the same error family as splitting by slice**, one stage later in the
pipeline.

A synthetic test with a per-patient random effect makes the gap stark: the
patient-level interval is ~5× wider than the slice-level one.

### What that immediately explains

Validation AUROC peaked at **0.909**; test AUROC was **0.768**. A tempting
story ("we overfit the validation set") is unnecessary — 0.909 sits *inside*
the test CI of [0.651, 0.936]. The gap is noise.

Likewise, an earlier 2-epoch run scored test AUROC 0.805, *higher* than the
15-epoch run's 0.768. **I did not switch to 2 epochs.** Choosing the epoch
count by test score is selecting on test, which converts the test split into a
training signal and makes the reported number meaningless. The two scores are
indistinguishable given the interval anyway. This is exactly the caveat the
Step 9 dataset card recorded: *~12 test patients means wide confidence
intervals; treat small differences as noise.*

---

## 4. Discipline enforced in code

- **Test split touched exactly once**, at the end, with the model and threshold
  already chosen on validation.
- **Model selection on validation AUROC** (threshold-free, so model choice is
  not entangled with threshold choice).
- **Threshold tuned on validation** — 0.5 is arbitrary under imbalance, and the
  selected value was **0.458**. A fixed 0.5 cut gave F1 = 0.000 in several
  epochs.
- **Imbalance handled by weighted loss** (`pos_weight = 6.74 = neg/pos`), not by
  discarding negatives.
- **Normalization fitted on the train split only.** Computing mean/std over all
  splits is a small leak, and the kind nobody notices. A test asserts the
  train-only mean *differs* from the all-splits mean, so the test would fail if
  the code ever leaked.
- **Augmentation: horizontal flip only.** A head CT is roughly left-right
  symmetric and "is there a hemorrhage anywhere in this slice" is unchanged by
  mirroring. Vertical flips would produce anatomically impossible images.

---

## 5. The serious bug: a test corrupted production data

`python -m medimageforge train` failed with:

```text
KeyError: "Cannot resolve pseudonym 'PAT-98c32af848a5' — wrong manifest?"
```

The `patients` table held `PAT-c835a293b1a8` for patient 049, but the release
expected `PAT-98c32af848a5`. Cause:

```python
# tests/test_privacy.py  (the old version)
def test_real_patients_are_all_mapped():
    db = data_path(config, "manifest_db")      # the REAL manifest
    mapping = build_pseudonym_map(db, SALT)    # ...and this WRITES
```

`build_pseudonym_map` persists what it computes, so **running `pytest`
overwrote the real pseudonyms with test-salt values** and left the published
release unable to resolve its own image paths.

### The second, worse defect

Step 9's `release --verify` had reported **PASS** on this broken state —
because it *also* called `build_pseudonym_map`, which recomputed the mapping
with the correct salt and **overwrote the corruption before checking it**. The
verification was silently repairing the damage it was supposed to detect.

### The fixes

1. **Split read from write.** New `read_pseudonym_map()` never writes; only the
   privacy command calls the writing version. `verify_release` now reads.
   With that change verification correctly reported **5001 unresolvable
   pseudonyms** instead of PASS.
2. **Tests operate on a copy** of the manifest (`shutil.copy` into `tmp_path`).
3. Two new tests pin the behaviour: `read_pseudonym_map` leaves the database
   **byte-for-byte unchanged**, and the full suite now runs without breaking
   `release --verify`.

**Three lessons:**
- A test must never mutate production artifacts.
- A function named like a getter must not write. `build_*` that persists is a
  trap for every future caller.
- **A verifier that can repair cannot verify.** If a check fixes state before
  asserting, it will always pass.

Notably, it was Step 10's *read-only* loader that exposed this — the stricter
component surfaced a latent fault in an older one.

---

## 6. Smaller issue found

`float(loss)` on a grad-tracking tensor emitted a PyTorch warning and keeps the
autograd graph alive; replaced with `loss.item()`, which detaches first.

---

## 7. The run record

`artifacts/runs/<run-id>/run.json` answers "what produced this number?":

| field | why it is there |
|---|---|
| `dataset_version` + `dataset_index_sha256` | the data, pinned — not a folder path |
| `code_version`, `git_commit` | the code, pinned |
| `config`, `seed` | the knobs |
| `environment` (python, torch, platform, normalization stats) | the context |
| `baseline` | the number that had to be beaten |
| `history`, `selected_epoch`, `selected_threshold` | how the model was chosen |
| `metrics` incl. `auroc_ci` | the result, with honest uncertainty |

Alongside it: `model.pt` (weights) and `predictions.npz` (per-slice scores
**with patient ids**) — the patient ids are what make patient-level
bootstrapping and Step 11's error analysis possible.

---

## 8. Verified

```text
dataset version: v1.0    code 0.1.0 (e69057a4)
model: SliceCNN (97,521 parameters)     splits: 1750 / 393 / 358
selected epoch 10, threshold 0.458
test AUROC 0.7681  95% CI [0.651, 0.936]  (12 patients)
Beats the trivial baseline: YES          duration 70 s
```

- Reproducible: two runs with seed 20260915 produced identical per-epoch losses
  and identical test AUROC
- `pytest` → **154 passed**
- `release --verify` still PASSES after a full test run (the regression that
  started this section)

## 9. Concepts learned

- **Train from a versioned release**, so the experiment record is checkable.
- **Accuracy is the wrong headline metric under imbalance** — it can fall while
  the model improves dramatically.
- **Ties matter** in a hand-written AUROC; average ranks are what make an
  uninformative model score 0.5.
- **Bootstrap over patients, not slices** — otherwise you report false
  precision.
- **Never select anything on the test split**, including the epoch count, even
  when a different choice would look better.
- **Readers must not write**, and **a verifier that repairs cannot verify**.
