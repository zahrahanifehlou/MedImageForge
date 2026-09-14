# Step 4 — Ingestion pipeline & manifest

## What we built

`src/medimageforge/manifest.py` plus an `ingest` subcommand:

```bash
python -m medimageforge ingest       # run it twice — the second run proves idempotency
```

It walks `data/`, fingerprints every file with SHA-256, and upserts one row
per file into `artifacts/manifest.db` (SQLite).

## Why it exists

The filesystem can *store* files but cannot *answer questions*: "which files
belong to patient 049?", "what changed since the last run?", "which slices
passed curation?". The manifest is the platform's **source of truth** — you
cannot manage what you cannot list. Every later step (curation, labels, QC,
versioning) reads and updates these rows instead of re-walking the disk.

## The schema

| Column | Purpose |
|---|---|
| `rel_path` | path relative to `data_dir` — the file's stable identity (UNIQUE) |
| `kind` | `slice` / `mask` / `metadata` |
| `patient_id`, `window`, `slice_no` | parsed from the path; NULL for metadata |
| `sha256`, `size_bytes` | content fingerprint — how we detect change |
| `status` | `raw` today; `curated`/`rejected` in Step 5 |
| `first_seen`, `last_seen` | ISO-8601 UTC — the beginning of the audit trail |

`rel_path` (not an absolute path) is the identity on purpose: the manifest
stays valid if the repo moves or runs in a container.

## How ingest works

```text
for every file under data/:
    compute sha256 + size
    row absent           → INSERT            → "new"
    same sha256 + size   → touch last_seen   → "unchanged"
    fingerprint differs  → UPDATE in place   → "updated"
rows whose file vanished → status='missing'  (never DELETE)
```

## Concepts learned

- **Idempotency.** Run 1: 5326 new. Run 2: 5326 unchanged, 0 new. A pipeline
  you can safely re-run after a crash never produces duplicate or
  half-applied state — this is *the* property that makes automation trustable.
- **Content-addressed change detection.** We compare hashes, not timestamps.
  A restored backup or a re-copied file has a new mtime but identical bytes —
  hashing is what makes "unchanged" actually mean unchanged.
- **A manifest records history; it does not rewrite it.** A file that
  disappears is flagged `status='missing'`, not deleted. In a regulated
  system, "we used to have this file" is itself evidence.
- **Why SQLite over CSV.** Indexed queries, atomic transactions, and in-place
  status updates. Same job as RDS/DynamoDB in Phase 6 — different substrate.

## Verified

- Two consecutive runs: 5326 new → 5326 unchanged (idempotent)
- Manifest counts match the Step 2 explorer exactly: 82 patients,
  2501 brain slices, 2500 bone slices, 318 masks, 7 metadata files
- `pytest` → 21 passed, including synthetic tests for content change and
  vanished files (cases the real dataset never shows)
