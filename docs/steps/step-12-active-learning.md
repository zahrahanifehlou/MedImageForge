# Step 12 — Active learning loop

```bash
python -m medimageforge active-learning                        # 3 seeds (config default)
python -m medimageforge active-learning --seeds 8              # 8 seeds x 3 arms = 24 trainings
python -m medimageforge active-learning --from-report --publish # publish v1.1 without retraining
```

This closes Phase 4 and the platform's central loop:
**Data → Model → Errors → Better data → Better model.**

---

## 1. What & why

Annotation is the scarce resource in medical imaging. A radiologist's hour is finite, so
the question is not "label more" but **"label WHICH cases"**. Active learning's answer:
the cases the current model is most unsure about, since a case it already classifies
confidently teaches it little.

## 2. How you test that on an already-labelled dataset

CT-ICH is fully annotated, so the workflow is **simulated**:

```text
57 train patients
   ├── 40% seed pool (23 patients)  ──► train round-0 model
   └── 60% "unlabelled" pool (34)   ──► score with round-0 model
                                          │
                        rank patients by uncertainty
                                          │
                    select 12 ──► "annotate" = reveal existing label
                                          │
                              retrain ──► compare on the SAME test split
```

The simulated reviewer is a **perfect, instant oracle** — the one unrealistic element,
and it flatters active learning if anything (a real annotator is slower and fallible).

---

## 3. The design decision that makes or breaks the experiment

**A random-selection control is mandatory.** Adding data almost always helps, so
"the model improved after annotation" is *not* evidence for active learning. The claim
under test is narrower:

> uncertainty sampling beats **random** sampling at the **same** annotation budget.

So every seed trains three models:

| arm | training pool |
|---|---|
| `seed` | round 0: the 23-patient seed pool only |
| `uncertainty` | seed pool + 12 most-uncertain patients |
| `random` | seed pool + 12 randomly chosen patients ← **the control** |

Validation and test are **byte-identical in every arm** (verified by test and by
comparing v1.0 and v1.1 indices), which is what makes the comparison valid.

Other choices, and why:

- **Selection per patient, not per slice.** A radiologist annotates a study. It also
  keeps uncertainty honest: if some slices of a patient were already in training, the
  model has seen that head and its uncertainty on the rest is not representative.
- **Binary predictive entropy**, aggregated per patient by mean. For two classes,
  entropy, least-confidence and margin sampling induce the *same* ranking (a test pins
  this); entropy is used because it generalizes unchanged to more classes.
- **Normalization loaded from the round-0 run record.** Re-deriving it would score the
  pool with different preprocessing than the model was trained with, making the whole
  ranking meaningless.
- **Seed pool stratified on hemorrhage.** An accidentally all-negative round-0 model
  would produce a meaningless uncertainty ranking.

---

## 4. The result — and the mistake I almost published

### After 3 seeds

```text
uncertainty vs random:  deltas [+0.054, +0.111, -0.030]
                        mean +0.045 AUROC, wins 2 / losses 1
```

That looks like active learning working. **It was noise.**

### After 8 seeds

| arm | mean test AUROC | std | min | max |
|---|---|---|---|---|
| seed (round 0) | 0.7903 | 0.0605 | 0.6847 | 0.8886 |
| uncertainty | 0.7897 | **0.0417** | 0.7048 | 0.8360 |
| random | 0.7956 | 0.0709 | 0.6701 | 0.8759 |

```text
uncertainty vs random: deltas [+0.054, +0.111, -0.030, -0.147,
                               -0.066, +0.055, +0.023, -0.048]
    mean -0.0059   95% CI [-0.0750, +0.0633]   4 wins / 4 losses
uncertainty vs seed:   mean -0.0006   95% CI [-0.0681, +0.0670]   4W/4L
random vs seed:        mean +0.0053   95% CI [-0.0465, +0.0571]   5W/3L
```

**A perfect coin flip.** Three honest conclusions:

1. **This dataset cannot distinguish uncertainty sampling from random selection.** The
   experiment could only have detected an effect of **≥ 0.085 AUROC**; no plausible
   active-learning gain is that large. This is a statement about **statistical power**,
   not proof that active learning fails.
2. **Adding 52% more training data (23 → 35 patients) produced no measurable gain
   either** (`uncertainty vs seed` and `random vs seed` both null). With a 12-patient
   test split whose per-run CI spans ±0.15, the noise floor is simply above the effect.
3. The one suggestive signal is **lower variance** for uncertainty (std 0.042 vs 0.071) —
   more consistent across seeds. With 8 seeds a variance estimate is itself very noisy,
   so this is a hypothesis, not a finding.

### Why this is the most valuable lesson in the step

Stopping at 3 seeds would have produced a confident, wrong claim. The guard is not
"collect more data until it works" — it is **reporting an interval**. Even the 3-seed
result had a CI of **[−0.131, +0.222]**, which already said "we cannot tell":

```python
paired_statistics(deltas[:3])
# mean_delta 0.0453, ci [-0.1312, 0.2218], significant: False
```

So `summarize` now reports the paired CI, win/loss counts and the **minimum detectable
effect** on every comparison, and `significant` is only true when the interval excludes
zero. A mean alone invites stopping as soon as the number looks good — which is how a
coin flip becomes a finding.

This is the same discipline as Step 8 (calibrate against a baseline), Step 10 (bootstrap
over patients) and Step 11 (report measurability): **state the uncertainty of your own
conclusion.**

---

## 5. What the strategies actually picked (seed 0)

| strategy | patients | with hemorrhage | overlap with the other |
|---|---|---|---|
| uncertainty | 12 | 6 | 5 |
| random | 12 | 7 | 5 |

The two strategies overlapped on 5 of 12 patients — with only 34 candidates, a "smart"
selection and a random one are substantially the same set, which is part of why the
effect is undetectable here.

---

## 6. `datasets/v1.1`

Published from seed 0's uncertainty-selected pool — the dataset a team would adopt after
one annotation round:

| | v1.0 | v1.1 |
|---|---|---|
| train patients | 57 | **35** |
| train brain slices | 1750 | 1049 |
| validation / test | — | **byte-identical to v1.0** |

`metadata.json` records `derived_from: v1.0` and the full selection provenance
(strategy, budget, the 12 patient pseudonyms, experiment seed). The dataset card states
the honest caveats: the pool is a *subset* of v1.0, it was selected by model uncertainty
so it is **deliberately not a random sample**, and the Step 11 subdural blind spot
persists.

Note v1.1 has *fewer* training patients than v1.0 — it is a simulated first annotation
round, not a superset. Calling it "v1.1" follows the roadmap's naming.

---

## 7. Bugs found

1. **`NameError: name 'pd' is not defined`** in the publish path. `cli.py` never imported
   pandas, and the publish branch only runs with `--publish`, which I had not exercised
   until the final run — so 24 trainings completed and *then* it crashed. Same family as
   the Step 5 SQL-binding bug: **a code path that only runs under a flag is untested by
   default.**
2. That crash exposed a design flaw: publishing a release required recomputing an
   8-minute experiment. Added `--from-report` so the saved `experiment.json` can be
   republished directly. **An expensive computation and the artifact it produces should
   be separable.**

---

## 8. Verified

- 24 trainings across 8 seeds × 3 arms; every arm evaluated on identical val/test data
- v1.1 published; validation and test confirmed **identical** to v1.0; no patient in two
  splits; full selection provenance recorded
- `pytest` → **215 passed** (28 new), including: entropy properties, stratified pool is a
  true partition, random control differs from the strategy, restricting train leaves
  val/test untouched, and losses are reported rather than hidden
- `release --verify` still PASSES after the full suite

## 9. Concepts learned

- **The control arm is the experiment.** Without random selection at equal budget, an
  active-learning result is unfalsifiable.
- **Pair by seed.** Between-seed variance here (std ~0.06) exceeds the effect sought.
- **Report intervals and minimum detectable effect**, not means — otherwise the stopping
  rule becomes the finding.
- **A null result can be a power problem.** Saying which of the two it is requires
  computing what you *could* have detected.
- **Simulated oracles flatter the method**; say so explicitly.
- **Flag-gated code paths are untested code paths.**
