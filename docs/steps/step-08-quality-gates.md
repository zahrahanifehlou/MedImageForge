# Step 8 — Automated quality gates

```bash
python -m medimageforge qc                 # exit 0 = PASS, 1 = FAIL
python -m medimageforge qc --skip-leakage  # skip the slow perceptual-hash check
```

---

## 1. What & why

Steps 4–7 established what we **have**. QC asks whether it is internally
consistent enough to train on. *A dataset must earn its way to training* —
gates are the difference between a data lake and a data swamp.

**Severity, reusing Step 5's vocabulary:**

| | Meaning | Effect |
|---|---|---|
| **ERROR** | would corrupt training or invalidate results | gate **FAILS** (exit 1) |
| **WARNING** | real and noteworthy, but the data stays usable | gate passes, finding recorded |

## 2. The five gates

| Gate | Asks | Severity of a hit |
|---|---|---|
| `slice-counts` | do brain/bone counts agree per patient? | WARNING |
| `orphans` | labels ↔ images ↔ masks, in **both** directions | ERROR |
| `mask-label-agreement` | does a mask exist exactly when a hemorrhage is labelled? | ERROR |
| `demographics` | coverage, plausible values, patient-vs-slice label agreement | ERROR / WARNING |
| `leakage` | is the same scan filed under two patient IDs? | ERROR |

---

## 3. Results — two real inconsistencies found

```text
[WARN] slice-counts    084: brain=36, bone=35
[PASS] orphans
[PASS] mask-label-agreement
[WARN] demographics    050: age 0.5833 | 066: age 0.6 | 085: age 0.0027
[PASS] leakage         109 candidates, 0 confirmed
Errors: 0   Warnings: 4        GATE: PASS
```

1. **Patient 084 has 36 brain slices but only 35 bone slices** — brain slice
   36 has no bone counterpart. The roadmap predicted this dataset contains a
   real inconsistency; this is it.
2. **Three paediatric patients**, including patient 085 at **0.0027 years
   (~1 day old)**.

Both are genuine data, so both are warnings. The gate passes honestly.

Strong cross-validations that came back clean:

- every label row has its image, every image has its label (0 orphans)
- **318 masks ↔ 318 hemorrhage-positive slices**, exact in both directions
- the two CSVs agree: patient-level labels derived from slices match the
  demographics summary in **0** of 82 patients — which also *validates the
  assumption* that blank cells there mean 0

---

## 4. Leakage detection — the centrepiece

### Why it matters
Step 9 will split **by patient**. That is worthless if the same scan exists
under two patient IDs: it lands in train *and* test, and the reported accuracy
becomes fiction.

### Why SHA-256 is not enough
Step 4 already found **0 exact duplicates**. But a re-compressed copy has
completely different bytes and identical appearance. Proven by test:

| image type | SHA-256 equal after re-compression? | dHash distance |
|---|---|---|
| a real CT slice | no | **0** |

So we need a *perceptual* hash. `dhash` resizes to 9×8 and compares each pixel
with its right neighbour — 64 bits describing gradient structure.

### Why it is two-stage

```text
stage 1  dhash all 2501 slices, compare all ~3.1M pairs   ~6 s
         └─ one matrix multiply: encoding bits as ±1 makes
            the dot product equal (64 − 2·distance)
stage 2  load real pixels for the 109 cross-patient
         candidates only, score correlation                instant
```

Loading 2501 full images to compare everything would be wasteful; loading the
~100 that stage 1 proposes costs nothing. Cheap filter, expensive confirmation.

### Why the threshold is **calibrated**, not guessed

This is the part that matters. Measured on this dataset:

| | correlation |
|---|---|
| closest **cross-patient** pair (`070/29` vs `114/25`) | **0.950** |
| same-patient **adjacent** slices | median **0.896**, max 0.994 |

**The distributions overlap.** Head CTs are anatomically stereotyped — two
different patients at the same level look *more* alike than two consecutive
slices of one patient. A threshold chosen by intuition (say 0.9) would flag
hundreds of distinct patients as duplicates.

I verified the closest pair visually: **different skull shape, different
scalp layers, different ventricle morphology — two different people** at the
same anatomical level. Its pixel correlation (0.848 at full resolution) is
*lower* than that of adjacent slices of one patient (0.965).

So `leakage_min_correlation: 0.99` sits above the entire observed
cross-patient range, and the CLI still prints the top candidates **for human
review** — a gate is a decision, not the whole picture.

---

## 5. Why IQR outlier detection was the wrong tool

The obvious way to find demographic outliers is the interquartile rule. On
this dataset:

```text
Q1 = 11.25   Q3 = 40.0   IQR = 28.75
outlier bounds -> [-31.9, 83.1] years      IQR outliers found: 0
```

It does not flag the **1-day-old neonate** at all, because the age
distribution is so wide that the bounds become meaningless (a *negative*
lower bound for age). A statistical rule knows nothing about the domain; the
domain rule "age < 1 year is clinically notable" catches it immediately.

**Lesson:** generic statistics are not a substitute for domain knowledge.

---

## 6. Two bugs found while testing

### A fixture that did not resemble the data
The dHash tests initially used `rng.integers(...)` **random noise** and
failed. Noise is the worst case for a perceptual hash: every adjacent-pixel
comparison is a coin flip that JPEG easily flips. Measured:

| fixture | dHash distance under re-compression |
|---|---|
| pure random noise | 4 (unstable) |
| smooth blobs | 1 |
| real CT slice | **0** |

The check was correct; the fixture had nothing in common with a CT scan. It
now generates smooth Gaussian blobs. **A fixture must share the statistical
properties of the real data or it tests the wrong thing.**

### Rounding before a threshold comparison
`find_leakage` rounded correlation to 4 decimals *before* testing it, so
`0.999955` became `1.0` and passed a `0.999999` gate. Full precision is now
kept for the comparison and rounding applied only for display.

**Never round a value before comparing it against a threshold.**

---

## 7. Verified

- Real dataset: **0 errors, 4 warnings, GATE: PASS** (exit 0)
- Caught the real inconsistency the roadmap promised (patient 084)
- Leakage: 109 candidates, 0 confirmed — no duplicate scans across patients
- `pytest` → **92 passed**, covering every gate against deliberately broken
  data, a genuine cross-patient duplicate, and the gate's severity semantics

## 8. Concepts learned

- **Gates return exit codes** — a check nobody is forced to honour is decoration.
- **ERROR vs WARNING** — real anomalies must not silently discard usable data.
- **Perceptual vs cryptographic hashing** — different questions: "are these
  the same bytes?" vs "are these the same picture?".
- **Cheap filter, expensive confirmation** — the shape of most scalable
  duplicate detection.
- **Calibrate thresholds against a within-class baseline** — otherwise you
  cannot tell "similar" from "identical".
- **Domain rules beat generic statistics** for plausibility checks.
