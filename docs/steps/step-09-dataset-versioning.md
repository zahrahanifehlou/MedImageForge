# Step 9 — Dataset versioning

```bash
python -m medimageforge release            # publish datasets/v1.0 (refuses to overwrite)
python -m medimageforge release --verify   # check checksums, drift, and privacy
python -m medimageforge release --force    # deliberately replace a release
```

This closes Phase 3.

---

## 1. What & why

**"Which data trained this model?" must always be answerable.** A folder path
is not an answer, because folders change. A *version* is an answer — so a
release is an immutable snapshot containing the split, an index of every file,
machine-readable provenance, checksums, and a dataset card.

---

## 2. The one rule that matters most: split by PATIENT, never by slice

Each patient contributes ~30 slices of the same head, and Step 8 measured
adjacent slices at **correlation 0.896**. Split by slice and slice 14 lands in
train while slice 15 lands in test — the model is then evaluated on data it has
effectively already seen. The score looks excellent and means nothing.

```text
WRONG (by slice)                     RIGHT (by patient)
patient 049 ─┬─ slice 14 → train     patient 049 ── all slices → train
             └─ slice 15 → test      patient 114 ── all slices → test
   leakage: same head in both            no overlap possible
```

It is enforced, not hoped for: `assert_no_patient_overlap` runs on every
release, and a test proves the detector actually raises.

Step 8's leakage gate is what makes this trustworthy: it confirmed **no
duplicate scan exists under two patient IDs**, so patient-level separation
really does separate the data.

---

## 3. Stratification, and the limit we hit

Only 36 of 82 patients have a hemorrhage, and the test split holds ~12
patients. Random splitting could easily land 2 or 9 positive patients in test.
So ratios are applied **within each stratum**.

**But stratification is limited by your smallest stratum.** Measured:

| stratum (hemorrhage × fracture) | patients |
|---|---|
| no-HGE, no-fracture | 45 |
| HGE, fracture | 21 |
| HGE, no-fracture | 15 |
| **no-HGE, fracture** | **1** |

A stratum of one **cannot be divided across three splits**. So we stratify on
hemorrhage only (36/46) and *report* fracture, age, and gender balance in the
dataset card instead of forcing it. Singleton strata degrade predictably —
`round(1 × 0.70) == 1` sends the lone patient to train — and a test pins that.

### Patient balance does not imply slice balance

| split | patients | w/ HGE | brain slices | HGE slices | rate |
|---|---|---|---|---|---|
| train | 57 | 25 | 1750 | 226 | 0.129 |
| validation | 13 | 6 | 393 | 52 | 0.132 |
| test | 12 | 5 | 358 | 40 | 0.112 |

A hemorrhage-positive patient contributes **between 1 and 19 positive slices**
(median 9). Balancing patients therefore cannot balance slices — which is why
the card *reports* the rates rather than promising them.

---

## 4. A release stores pointers plus hashes, not copies

Copying 333 MB per version does not scale, and versions mostly overlap. Each
index row records the file's path **and its sha256**:

```csv
patient,split,window,slice_no,path,sha256,...,hemorrhage
PAT-98c32af848a5,train,bone,1,PAT-98c32af848a5/bone/1.png,fa4c6ca0...,0
```

If a curated file is ever modified, `--verify` fails and we know the release no
longer describes reality. This is the pointer-plus-hash idea behind DVC and
git-annex. A test proves drift is caught: overwrite one curated PNG and
verification reports `content-changed`.

## 5. Immutability, enforced three ways

1. `release` **refuses** to overwrite an existing version (`--force` required).
2. Every file is chmod **0444** after writing — a later pipeline cannot
   silently edit a published release. Verified by test.
3. `CHECKSUMS.txt` fingerprints the release's own files, so tampering is
   detectable even if permissions are changed.

---

## 6. A privacy regression I caught by reading my own artifact

The first `index.csv` contained:

```csv
curated_path
049/bone/1.png        ← the REAL patient number
```

That directly contradicts Step 6: a release is the artifact that **leaves the
controlled zone**, so it must carry pseudonyms only. The fix is architectural
rather than cosmetic — files are referenced by *pseudonymous* path:

```csv
path
PAT-98c32af848a5/bone/1.png
```

The curated file still lives at `curated/049/bone/1.png`. Resolving the
pseudonymous path requires the manifest's `patients` table — so **verification
only works from inside the controlled zone**, which is the re-identification
control working exactly as designed. `--verify` now also scans the index for
real patient numbers and reports `real-patient-id-in-index`.

**Lesson:** privacy rules decay unless each new artifact is checked against
them. Read your own outputs.

---

## 7. The reproducibility bug — the most important lesson in this step

After fixing the paths I republished, and the test split **changed**: 37
hemorrhage slices became 54, with an unchanged seed. The cause:

```python
rng = np.random.default_rng([seed, hash(str(key)) % (2**32)])   # BROKEN
```

**Python's `hash()` for `str` is salted per process** (`PYTHONHASHSEED`). Every
run derived a different sub-seed, so "seed 20260915" meant nothing across
processes.

My determinism test had passed — because it called `split_patients` twice
**inside one process**, where the salt is constant. It gave false confidence
about the single property this step exists to provide.

The fix is a process-independent digest:

```python
def _stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
```

Verified the way it should have been tested from the start — across separate
processes, and under forced hash seeds:

```text
run 1                 51ae3999ce06269c68ed4df8
run 2                 51ae3999ce06269c68ed4df8
run 3                 51ae3999ce06269c68ed4df8
PYTHONHASHSEED=1      51ae3999ce06269c68ed4df8
PYTHONHASHSEED=9999   51ae3999ce06269c68ed4df8
```

and republishing twice now yields a **byte-identical** `index.csv`.

`test_stable_hash_is_pinned_to_known_values` hardcodes the expected outputs, so
if anyone reintroduces `hash()` the suite fails immediately instead of silently
changing every future split.

**Two lessons:**
- **Reproducibility must be verified across processes**, not within one. A
  determinism test that runs twice in the same interpreter can pass while the
  property is broken.
- **Never derive a random seed from `hash()`** on a string.

---

## 8. The dataset card

`datasets/v1.0/dataset_card.md` records source and label-file **sha256**,
per-split counts, how the split was made (unit, stratification, ratios, seed),
the privacy model, and **known issues** carried forward from Step 8 so a reader
does not have to dig through our artifacts:

- patient 084's missing bone counterpart
- three patients under 1 year old (youngest ~1 day)
- ~13% class imbalance
- masks thresholded from JPEG, so boundaries are approximate
- consensus labels only, so annotation uncertainty cannot be estimated

It also states the honest limitation: **~12 test patients means wide confidence
intervals** — report patient-level metrics and treat small differences as noise.
That matters directly for Steps 10–12.

---

## 9. Verified

| Check | Result |
|---|---|
| `release` | 57 / 13 / 12 patients; 25 / 6 / 5 hemorrhage-positive; 5001 images |
| Patient overlap | none (enforced + tested) |
| Reproducibility | identical across processes and hash seeds; byte-identical index |
| Immutability | re-run refused; files mode 444 |
| `--verify` | checksums 0, drift 0, privacy leaks 0 → PASS |
| `pytest` | **113 passed** |

## 10. Concepts learned

- **Split by patient** — the single most important anti-leakage rule in
  medical imaging, and it must be *enforced*, not assumed.
- **Stratification is bounded by the smallest stratum**; balance what you can,
  report the rest.
- **Patient-level balance ≠ slice-level balance.**
- **Pointer + hash beats copying** — versions stay cheap and drift becomes
  detectable.
- **Immutability needs mechanism** (refusal, permissions, checksums), not
  good intentions.
- **Reproducibility is a cross-process property** — and the reason to distrust
  `hash()` anywhere near a seed.
