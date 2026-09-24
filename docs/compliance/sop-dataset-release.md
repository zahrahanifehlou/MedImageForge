# SOP-02 — Dataset release & verification

| | |
|---|---|
| **Purpose** | Publish an immutable, patient-level, stratified dataset snapshot that a model run can cite forever. |
| **Scope** | QC gate → `release` → `release --verify`. |
| **Roles** | Data steward (runs), reviewer (approves the dataset card before use) |
| **Preconditions** | SOP-01 complete; QC report has `gate: PASS` |

## Procedure

1. **Run the QC gate:**
   ```bash
   python -m medimageforge qc
   ```
   `gate: FAIL` blocks release — fix the data, never the gate. Warnings are
   reviewed and acknowledged in the release notes.
2. **Publish** (bump `release.version` in `configs/default.yaml` first):
   ```bash
   python -m medimageforge release
   ```
   The command refuses to overwrite an existing version — **immutability is
   enforced, not requested.**
3. **Verify immediately:**
   ```bash
   python -m medimageforge release --verify
   ```
   Must print `VERIFY: PASS` (0 modified, 0 drifted, 0 real IDs leaked).
4. **Review the generated `datasets/vX.Y/dataset_card.md`** — split table,
   stratification basis, seed, labels sha256. The reviewer signs off here,
   not on the model.

## Verification

- `release --verify` PASS recorded in the audit log.
- `datasets/vX.Y/` contains `index.csv`, `splits.csv`, `CHECKSUMS.txt`,
  `metadata.json`, `dataset_card.md`.

## Records produced

- `datasets/vX.Y/` (immutable snapshot, self-contained provenance)
- `artifacts/audit.log` entries for `qc`, `release`, `release --verify`

## Rules that are enforced, not advisory

- Splits are **per patient** — a patient's slices can never straddle splits.
- Stratification on patient-level hemorrhage only (rare strata documented
  in the card rather than enforced — see Step 9).
- A version, once published, is never edited; corrections are a new version.
