# Model card — hemorrhage slice classifier (baseline CNN)

| field | value |
|---|---|
| Model | small CNN, PyTorch (`src/medimageforge/model.py`) |
| Task | binary classification: hemorrhage present on a 128×128 brain-window CT slice |
| Trained on | `datasets/v1.1` train split (35 patients, 1049 brain slices, 132 positive) |
| Evaluated on | `datasets/v1.1` test split (12 patients, 358 slices, 40 positive) |
| Headline metric | AUROC 0.768 (bootstrap CI in the eval report) |
| Threshold | chosen on **validation**, recorded in the run record; never re-tuned on test |

## Intended use

**Learning project — not for clinical use.** Demonstrates the slice-level
detection stage of a hemorrhage triage pipeline. Useful signals: scan-level
triage via max-pooling (patient recall 1.00 on this test set) — but the test
set is 12 patients, far too small for any deployment claim.

## Known limitations (measured, not assumed)

- **Subdural blind spot:** 56 subdural slices in train, **zero** in
  validation/test — the model's subdural ability is *unmeasured*, not zero.
  Found by Step 11 error analysis.
- **Slice ≠ scan:** slice recall 0.425 vs patient recall 1.00 (max rule).
  Good triage, weak localization.
- **Small data:** per-patient AUROC moves in 0.029 steps; CIs of ±0.15 mean
  most single-number claims are noise. Reported honestly in eval reports.
- **High sensitivity is expensive:** recall 0.95 requires flagging ~70% of
  all slices (precision 0.145).

## Provenance

- Dataset: PhysioNet CT-ICH (Hssayeni et al.), releases `v1.0`/`v1.1`
  (immutable, checksummed, patient-level splits, seed 20260915)
- Run record: `artifacts/runs/<run_id>/` — config, git commit, dataset
  version, weights, predictions
- Evaluation: `artifacts/eval/<run>/` — metrics + error-analysis renders
- Training command audited: `artifacts/audit.log` (hash-chained)

## Ethical / privacy notes

- All patient identifiers pseudonymized at the platform boundary; the model
  never sees real IDs.
- Dataset is de-identified research data; no consent workflow applies, but
  the platform treats it as if it were PHI by design.
