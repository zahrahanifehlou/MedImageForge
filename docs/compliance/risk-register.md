# Risk register

Format: ID · description · likelihood (L) × impact (I) → rating · mitigation · status.
Ratings: L/I ∈ {low, med, high}; rating = worst-credible product.

| ID | Risk | L | I | Rating | Mitigation (implemented) | Status |
|---|---|---|---|---|---|---|
| R1 | Real patient ID leaks into a deliverable (path, index, response) | med | high | **high** | Pseudonym-only boundary at Step 6; API speaks pseudonyms only; tests scan *whole response bodies*; `release --verify` checks index for real IDs; cloud mapping removes raw access via IAM | Mitigated — two real leaks found & fixed (Step 9 index.csv, Step 14 path leak); residual: new output surfaces need the same test pattern |
| R2 | Tampered or accidental edit of history (audit log, releases) | low | high | med | Hash-chained append-only audit log + `audit --verify`; immutable releases + `release --verify`; Object Lock in cloud mapping | Mitigated — tamper demo verified; residual: local file can still be *deleted* wholesale (backup story is out of scope) |
| R3 | Train/test leakage inflates metrics | med | high | **high** | Patient-level splits enforced; QC two-stage duplicate detection (dhash + 0.99 pixel correlation); same-patient slices can never straddle splits | Mitigated — the 0.99 threshold was *calibrated* against the cross-patient max (0.950), documented in Step 8 |
| R4 | Model deployed beyond evidence | med | high | **high** | Model card lists measured limits (subdural blind spot, 12-patient test set, CI widths); no deployment pathway exists in the code | Mitigated by documentation + absence of serving-for-prediction code |
| R5 | Non-reproducible runs (silent config/seed drift) | med | med | med | Fixed seeds in config; resolved-config hash in every audit record; run records store full config + git commit | Mitigated — `load-labels` known to be non-byte-reproducible (timestamps), documented Step 13 |
| R6 | Dataset released with quality errors | med | med | med | QC gate blocks `release` on errors; warnings surfaced in report | Mitigated — current report: 0 errors, 4 documented warnings |
| R7 | Pseudonymization salt loss or exposure | low | high | med | Salt gitignored + never read into responses; cloud mapping: Secrets Manager + single-role grant | Partially mitigated — local backup of `.secrets/` is operator responsibility; documented in SOP |
| R8 | Active-learning conclusion overclaimed | low | med | low | Comparison reports paired bootstrap CI + min detectable effect; `significant` flag only when CI excludes 0 | Mitigated — Step 12 reports null result honestly |
| R9 | Dependency/supply-chain drift breaks reproducibility | med | med | med | Pinned `requirements.txt`; CI on clean checkout; Docker image builds deterministically | Mitigated — Step 16 index-url incident documented; residual: pytorch extra-index availability |
| R10 | Single-operator knowledge concentration | high | low | low | Step-by-step docs for all 18 steps; SOPs; this register | Mitigated by documentation |

## Accepted risks (conscious decisions)

- **R4 residual:** the platform *could* be pointed at real patients — mitigated
  by policy (learning project), not a technical block.
- **R2 residual:** wholesale deletion of `artifacts/` is recoverable only from
  external backup; hash chain detects edits, not disappearance.
- **R7 residual:** salt backup is manual. Loss = permanent de-identification
  (arguably privacy-positive); theft = re-identification risk.
