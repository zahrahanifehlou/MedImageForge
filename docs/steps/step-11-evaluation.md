# Step 11 — Evaluation & error analysis

```bash
python -m medimageforge evaluate                 # latest run, test split
python -m medimageforge evaluate --run <id> --split validation --top-n 12
```

Outputs `artifacts/eval/<run_id>/{evaluation.json, evaluation.md, hardest/*.png}`.

---

## 1. What & why

Step 10 produced one number: **test AUROC 0.768**. That number cannot say *which*
hemorrhages are missed, *whether* it was measured on enough data to mean anything, or
*which patient* is carrying it. **Aggregate metrics lie by omission.** This step
interrogates them.

Step 11 is **read-only with respect to the model**. It loads saved predictions and uses
the threshold the training run already chose on validation. Re-tuning anything during
"analysis" is how a test split quietly becomes a training signal.

---

## 2. Prerequisite: a gap that had to be fixed first

`predictions.npz` saved patient IDs but **not slice numbers**, so a prediction could not
be joined to per-subtype labels or named in a "hardest slices" list. Fixed by carrying
`slice_numbers` through `SliceDataset` → `predictions.npz` and retraining.

The retrain reproduced **identical** per-epoch losses and test AUROC — confirming the
change was purely additive and, incidentally, re-demonstrating Step 10's determinism.

Why not recover slice numbers from row order? Because that couples the analysis to the
loader's internal iteration order. `join_subtypes` now joins on `(patient, slice_no)` and
**raises if the row count changes**, so a silent duplicate-key explosion cannot corrupt
every downstream metric.

---

## 3. Finding 1 — most subtypes are not measurable on v1.0

| subtype | train | val | **test** | recall | measurable? |
|---|---|---|---|---|---|
| epidural | 98 | 42 | **33** | 0.485 | yes |
| intraparenchymal | 56 | 11 | 6 | 0.333 | yes |
| intraventricular | 18 | 1 | 5 | 0.200 | yes |
| subarachnoid | 15 | 2 | **1** | 1.000 | **NO** — 1 case, recall can only be 0 or 1 |
| subdural | 56 | **0** | **0** | n/a | **NO** — ability is UNKNOWN, not zero |

**The model has 56 subdural training slices and zero in validation or test.** Whether it
detects subdural hemorrhage at all is *unknown*. Printing "recall 0.00" there would be a
lie by formatting, so `per_subtype_recall` returns `recall: None` with
`measurable: false` and an explicit note.

This is a **dataset-design** finding, not a model finding: Step 9 stratified on
*any hemorrhage*, which cannot control the distribution of five subtypes across 12 test
patients. It goes straight into Step 12's v1.1 work and belongs in the dataset card.

Also visible: recall falls with rarity — 0.485 (epidural) → 0.333 → 0.200. The model is
weakest exactly where training data is thinnest.

---

## 4. Finding 2 — slice-level and scan-level performance disagree completely

A radiologist reads a **scan**, not a slice. Aggregating with the **max rule** (any
convincing slice escalates the study):

| unit | recall | precision | AUROC | 95% CI |
|---|---|---|---|---|
| slice | 0.425 | 0.258 | 0.768 | [0.648, 0.929] |
| **patient (max)** | **1.000** | 0.556 | 0.686 | [0.333, 1.000] |
| patient (top-3 mean) | 1.000 | 0.556 | 0.714 | [0.343, 1.000] |

**Slice recall is 0.425, but scan recall is 1.000** — every one of the 5 hemorrhage
patients has at least one slice above threshold. As a triage tool that flags studies for
review, this model misses **no patient**; as a slice localizer it misses more than half
the slices. Both numbers are true, and quoting only one misleads.

The caveat is equally important: patient AUROC rests on 5 positive × 7 negative =
**35 comparable pairs**, so it moves in steps of **0.029** and its CI spans
[0.333, 1.000]. Differences smaller than one step are not merely uncertain — they are
*unrepresentable*. `patient_level_report` reports that granularity itself.

---

## 5. Finding 3 — one patient carries the headline number

| patient | positive slices | share | detected | recall |
|---|---|---|---|---|
| PAT-c0e2a1d10ac9 | 16 | **40.0%** | 2 | **0.12** |
| PAT-6bd0a673aef8 | 9 | 22.5% | 6 | 0.67 |
| PAT-73a7754e4fc4 | 7 | 17.5% | 1 | **0.14** |
| PAT-4d636aca9ef7 | 4 | 10.0% | 4 | **1.00** |
| PAT-aaabb2bf9e3a | 4 | 10.0% | 4 | **1.00** |

Per-patient recall ranges from **0.12 to 1.00**. Leave-one-patient-out confirms the
dependence:

```text
drop PAT-c0e2a1d10ac9 -> AUROC 0.861  (+0.093)
drop PAT-35405a8b1e6b -> AUROC 0.796  (+0.028)
drop PAT-6bd0a673aef8 -> AUROC 0.742  (-0.026)
```

Removing one patient moves the headline from 0.768 to **0.861**. The reported score is
substantially a statement about that patient.

---

## 6. Finding 4 — the visual hypothesis was wrong (and the measurement proved it)

The worst miss (`PAT-c0e2a1d10ac9` slice 16, score **0.020**) rendered as a **thin
epidural sliver hugging the inner skull**, low-contrast against adjacent bright bone. The
natural hypothesis: small peripheral lesions are lost at 128×128.

**Measured, and false:**

| outcome | n | median mask area (650²) |
|---|---|---|
| MISSED | 23 | **2542 px** |
| DETECTED | 17 | 757 px |

Missed lesions are *larger*; area as a predictor of detection scores AUROC **0.217**
(inverse). Then the confounder appeared: `PAT-c0e2a1d10ac9` has a median mask of
**11096 px** — 9× everyone else — and a recall of 0.12. One patient with both huge
lesions and near-total misses manufactured the whole "big lesions get missed"
correlation.

> **With 5 positive patients, slice-level covariates are inseparable from patient
> identity.** Any tidy per-slice story ("large bleeds", "peripheral bleeds") is
> confounded by the cluster structure.

So the report deliberately ships **per-patient detection rates** rather than a covariate
analysis it cannot support — keeping the confounding visible instead of hiding it behind
a plausible narrative. This is the same discipline as Step 8's calibrated threshold and
Step 10's patient-level bootstrap: *the unit of analysis must match the unit of
independence.*

**Lesson: look at the pixels to generate hypotheses, then measure to kill them.** The
image was worth rendering — it produced a hypothesis in seconds — and the measurement
was worth running, because the hypothesis was wrong.

---

## 7. Finding 5 — the cost of high sensitivity, quantified

A missed bleed can be fatal; a false alarm costs a radiologist seconds. The F1-optimal
threshold treats those as equally bad. So we also report the strictest threshold reaching
90% recall:

```text
threshold 0.458 (F1-optimal): recall 0.425,  49 false alarms, 23 missed
threshold 0.025 (recall>=0.90): recall 0.950, 224 false alarms,  2 missed
```

To catch 95% of hemorrhage slices this model must flag **224 of 318 negatives — 70% of
everything**. Precision collapses to 0.145. That is the honest trade-off a clinician
must be shown, not a hidden consequence of a default 0.5.

---

## 8. Bug found

`render_hardest_cases` wrote **zero images** while logging "cannot resolve PAT-…".
`read_pseudonym_map` returns *real → pseudonym*; the renderer needs *pseudonym → real*.
I passed the map in the wrong direction and the `.get()` returned `None` for every case —
a silent, total failure that only looked like a warning.

Fixed by using `load_pseudonym_map` (which returns the direction the renderer wants) and
a test asserting one image per hardest case. **Two same-shaped dicts with opposite
meanings is an easy trap; a test that counts outputs catches it, a warning does not.**

---

## 9. What the report contains

| Section | Question it answers |
|---|---|
| slice level + CI | how good, and how uncertain |
| per subtype | what can even be measured |
| patient level (max / top-k) | how good per *scan* |
| per-patient contribution + recall | is one patient carrying the score |
| leave-one-patient-out | how much would it move |
| threshold sweep | what high sensitivity costs |
| hardest cases | **named** failures for human review |
| `hardest/*.png` | the pixels, with mask overlay |

The images are named `fn_PAT-…_slice016_score0.020.png` — self-describing, pseudonymous,
and resolved through the manifest so they stay inside the controlled zone.

---

## 10. Verified

- `evaluate` on the real run reproduces every finding above
- subdural correctly reported **NOT MEASURABLE**; `PAT-c0e2a1d10ac9` identified as
  dominant with a +0.093 leave-one-out delta
- 16 error-analysis images written; the worst miss inspected visually
- `pytest` → **187 passed** (33 new)
- `release --verify` still PASSES after the full suite (Step 10 regression guard)

## 11. Concepts learned

- **Report measurability, not just metrics** — "no test data" is not "recall 0".
- **The unit of analysis changes the conclusion** — slice recall 0.425 vs scan recall
  1.000, both correct.
- **State an estimate's granularity** — 35 pairs means AUROC moves in steps of 0.029.
- **Leave-one-out exposes fragile headlines** — one patient was worth 0.093 AUROC.
- **Cluster structure confounds covariate analysis** — measure before believing a story
  the pixels suggested.
- **Name your failures** — "patient X slice 16, score 0.020, epidural" is actionable;
  "recall 0.425" is not. This list is exactly Step 12's active-learning queue.
