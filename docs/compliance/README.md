# Compliance package — reviewer navigation

A reviewer should be able to audit our data's journey end-to-end from these
documents alone. Read in this order:

1. **[data-journey.md](data-journey.md)** — the narrative: one image's path
   from raw arrival to a model prediction, with the artifact that proves each
   hop.
2. **[audit-evidence.md](audit-evidence.md)** — real outputs from the
   platform's own verification tools (audit chain, release verify, QC gate,
   lineage trace). These are *evidence*, not claims.
3. **[dataset cards](../../datasets/v1.1/dataset_card.md)** — auto-generated
   per release (`datasets/*/dataset_card.md`), plus the model card in
   [model-card.md](model-card.md).
4. **[risk-register.md](risk-register.md)** — known risks, their mitigations,
   and which are accepted vs open.
5. **SOPs** — the operating procedures a team would follow:
   - [sop-data-ingestion.md](sop-data-ingestion.md)
   - [sop-dataset-release.md](sop-dataset-release.md)
   - [sop-annotation-review.md](sop-annotation-review.md)
   - [sop-audit-review.md](sop-audit-review.md)

## What each compliance need maps to

| Regulatory expectation | Where it's satisfied |
|---|---|
| Traceability (data → model) | `audit --trace`, release `metadata.json`, manifest `source_sha256` chain |
| Integrity | `release --verify`, `CHECKSUMS.txt`, `SHA256SUMS.txt`, hash-chained audit log |
| Privacy / de-identification | Step 6 privacy gate, pseudonym-only API, salt in `.secrets/` (gitignored) |
| Change control | immutable releases, audit log records every command + config hash |
| Quality management | QC gate blocks release on errors; findings documented in step docs |
| Risk management | risk-register.md |
| Operating procedures | sop-*.md |
