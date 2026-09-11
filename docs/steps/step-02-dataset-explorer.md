# Step 2 — Dataset explorer

## What we built

A read-only `explore` subcommand (`python -m medimageforge explore`) backed by
`src/medimageforge/explore.py`. It scans `data/Patients_CT`, parses both CSVs,
cross-checks labels against files, verifies every file against
`SHA256SUMS.txt`, and prints a report ending in a PASS/FAIL integrity gate.

## Why it exists

You cannot build a pipeline for data you have not measured. Every number a
later step asserts ("82 patients", "~2.5k slices") now comes from code, not
from the README. And checksums are the first job of ingestion: if bytes are
corrupt, every downstream result is meaningless, so we verify before trusting.

## How it works

```text
explore command
   ├─ scan_raw_dir()      walk Patients_CT → per-patient slice + mask counts
   │                      (masks *_HGE_Seg.jpg excluded from slice counts)
   ├─ load_labels()       pandas read_csv(encoding="utf-8-sig") — strips BOM
   ├─ load_demographics() same; normalizes multi-line quoted headers
   ├─ label_slice_exists() every label row → brain/<n>.jpg must exist
   └─ verify_checksums()  sha256 every file, compare with SHA256SUMS.txt
                          reports verified / mismatched / missing / extra
```

## Real measurements (this dataset)

| Metric | Value |
|---|---|
| Patients | 82 |
| Brain slices | 2501 |
| Bone slices | 2500 (one patient missing a bone slice — noted for Step 8 QC) |
| Segmentation masks | 318 |
| Label rows | 2501 — one per brain slice, all matched |
| Hemorrhage slices | ~318 of 2501 — heavy class imbalance (matters in Phase 4) |
| Checksums | 5325 verified, 0 mismatched, 0 missing, 1 extra |

The one "extra" file is `SHA256SUMS.txt` itself — a file cannot list its own
hash, so it is always unlisted. Expected, not an error.

## Concepts learned

- **Measure before you build.** The explorer is read-only — measuring changes
  nothing, which is what makes it safe to run anytime.
- **Integrity vs validity.** Checksums prove the bytes match what was
  published; they say nothing about whether the data is *correct*. That is
  curation/QC (Steps 5 and 8).
- **Messy real-world inputs:** UTF-8 BOM in the labels CSV, multi-line quoted
  headers in demographics, `049` on disk vs `49` in CSVs. Ingestion code earns
  its keep here.
- **Three distinct integrity failures** mean different things: `mismatched`
  (corruption), `missing` (incomplete copy), `extra` (unexpected files).

## Verified

- Report matches reality: 82 patients, 2501/2500 slices, all checksums pass
- `pytest` → 10 passed (including a synthetic test covering all three
  checksum failure modes)
