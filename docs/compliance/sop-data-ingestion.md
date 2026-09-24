# SOP-01 — Data ingestion & labeling

| | |
|---|---|
| **Purpose** | Bring new imaging data into the platform with integrity verification and a manifest record. |
| **Scope** | Raw arrival → manifest rows → label loading. Covers `ingest`, `load-labels`. |
| **Roles** | Data steward (runs, reviews), reviewer (spot-checks) |
| **Preconditions** | Raw data placed in `data/Patients_CT/` per naming spec; label CSV updated if applicable; publisher checksums present in `data/SHA256SUMS.txt` |

## Procedure

1. **Stage the data** in `data/Patients_CT/` — patient dirs zero-padded to 3
   digits, `brain/`/`bone/` windows, masks suffixed `_HGE_Seg`.
2. **Run ingestion:**
   ```bash
   python -m medimageforge ingest
   ```
   This re-verifies `SHA256SUMS.txt`, hashes every file, and upserts the
   manifest. A checksum mismatch **stops the run** — do not bypass; escalate.
3. **Load labels** (if the label CSV changed):
   ```bash
   python -m medimageforge load-labels
   ```
4. **Spot-check:**
   ```bash
   python -m medimageforge show-slice --patient 049 --slice 14
   ```
   Confirms the manifest, image decode, and mask join all agree.

## Verification

- `audit` lists the run with status `ok` and the expected config hash.
- `qc` (SOP-03) cross-checks labels↔images↔masks orphans.

## Records produced

- `artifacts/manifest.db` rows (slices, labels)
- `artifacts/audit.log` entry (actor, inputs fingerprinted before the run)

## Failure handling

| Symptom | Action |
|---|---|
| checksum mismatch | halt; quarantine the file; re-download from source |
| unexpected dimensions | file stays raw-only; `qc` will report it — do not hand-edit raw |
