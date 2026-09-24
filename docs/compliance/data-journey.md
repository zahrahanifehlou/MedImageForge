# The data journey — end-to-end traceability

Follow one image — say `Patients_CT/049/brain/14.jpg` — from arrival to
model prediction. Every hop leaves an artifact a reviewer can inspect.

## 1. Arrival → raw zone

The file lands in `data/Patients_CT/049/brain/`. Nothing modifies the raw
zone — the directory is read-only by convention and, in the cloud mapping,
by IAM.

**Evidence:** `data/SHA256SUMS.txt` — the publisher's official checksums;
`python -m medimageforge ingest` re-verifies them on every run.

## 2. Ingest → manifest

`ingest` walks the raw zone, hashes every file, and writes a row into
`artifacts/manifest.db` (slices table): real path, sha256, dimensions,
window. The manifest — not the filesystem — becomes the source of truth.

**Evidence:** `manifest.db` slices row for `049/14`; audit log entry for the
`ingest` run (actor, timestamp, config hash, input/output fingerprints).

## 3. Labels → annotation store

`load-labels` reads `data/hemorrhage_diagnosis.csv`, stores per-slice
hemorrhage subtype labels (plus the patient→mask join) in the manifest's
labels tables — keyed by **real** patient ID internally.

**Evidence:** labels tables; `hemorrhage_diagnosis.csv` sha256 recorded in
every release's `metadata.json`.

## 4. Curation → curated zone

`curate` validates each image (dimensions, decodability, window rules),
copies accepted files into `artifacts/curated/`, and records
`source_sha256` per curated file — a content-level link back to raw.

**Evidence:** `artifacts/curated/` tree; curation table rows; audit entry.

## 5. Privacy gate → de-id zone

`privacy` generates a salt (`.secrets/pseudonym_salt`, gitignored), maps
each real patient ID to `PAT-<hash>`, and exports demographics/label views
into `artifacts/deid/` with pseudonyms only. **From this point, any output
that may leave the platform uses pseudonyms.**

**Evidence:** `deid/` CSVs contain `PAT-…`, never `049`; the `patients`
mapping table stays inside the manifest (the controlled zone); API tests
assert no real ID in any response body — including inside paths.

## 6. QC gate

`qc` runs automated checks (slice counts, orphans, duplicates/leakage via
perceptual-hash + pixel-correlation, label sanity). Errors block release;
warnings are reported.

**Evidence:** `artifacts/qc_report.json` — gate `PASS`, 4 warnings
(documented window-count mismatches).

## 7. Release → immutable snapshot

`release` splits **patients** (not slices) into train/validation/test with a
fixed seed, writes `datasets/vX.Y/` — `index.csv`, `splits.csv`,
`CHECKSUMS.txt`, `metadata.json`, `dataset_card.md` — and refuses to
overwrite an existing version.

**Evidence:** `datasets/v1.1/` is self-contained; `release --verify`
re-hashes every referenced image and checks for real-ID leakage:
`VERIFY: PASS` (0 modified, 0 drifted, 0 leaked).

## 8. Train → run record

`train` reads *only* the release's index, trains, and writes
`artifacts/runs/<run_id>/` — config, metrics, weights, predictions,
threshold chosen on validation.

**Evidence:** run record JSON includes `dataset_version`, code version, git
commit, seed — the model is traceable to an exact dataset snapshot.

## 9. Evaluate → report

`evaluate` scores on the release's test split (never re-tuned), writes
`artifacts/eval/<run>/` — metrics with bootstrap CIs, per-patient detection,
worst-miss renders.

**Evidence:** eval report; the documented finding (subdural blind spot)
that motivated v1.1.

## 10. Every step → audit chain

Each command above appended a record to `artifacts/audit.log`: actor@host,
argv, timestamps, code version + git commit, config hash, input fingerprints
(before) / output fingerprints (after), status, `prev_hash`, `record_hash`.

**Evidence:** `audit --verify` → `INTACT`; `audit --trace artifacts/manifest.db`
reconstructs the producer chain automatically.

## The one-line summary for auditors

> Every artifact in `datasets/` and `artifacts/runs/` can be traced, through
> recorded hashes and the audit chain, back to specific bytes in the raw
> zone — and the platform can *prove* that chain hasn't been edited.
