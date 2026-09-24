# SOP-04 — Periodic audit & integrity review

| | |
|---|---|
| **Purpose** | Scheduled verification that the platform's history and artifacts are intact — the control that makes all other records trustworthy. |
| **Scope** | `audit --verify`, `audit --trace`, `release --verify`, QC re-run. |
| **Roles** | Data steward (runs), QA reviewer (reviews output, signs the review record) |
| **Cadence** | weekly + before every release + after any incident |

## Procedure

1. **Verify the audit chain:**
   ```bash
   python -m medimageforge audit --verify
   ```
   Expect `INTACT`. A `BROKEN` result gives the seq of the first mismatched
   record — that is an incident, escalate immediately; do not repair the log.
2. **Verify every release:**
   ```bash
   python -m medimageforge release --verify   # checks all versions present
   ```
   Expect `VERIFY: PASS` per version.
3. **Spot-trace an artifact** — pick one release or run record at random:
   ```bash
   python -m medimageforge audit --trace datasets/v1.1
   ```
   The chain should reach back to raw inputs or honestly report
   "no producing run" for pre-log artifacts.
4. **Re-run QC** on the live manifest:
   ```bash
   python -m medimageforge qc
   ```
5. **Record the review** — date, commands run, results, reviewer initials —
   appended to the QA review file (or, in the cloud mapping, filed against
   the Object-Locked audit bucket).

## Verification

This SOP *is* the verification. Its output is the review record.

## Escalation

| Finding | Severity | Action |
|---|---|---|
| `audit --verify` BROKEN | critical | freeze pipeline; incident review; the log localizes the first tampered record |
| `release --verify` FAIL | critical | quarantine the release; models trained on it flagged in run records via `dataset_version` |
| `qc` new errors | high | block release (automatic); investigate per SOP-01/02 |
| "no producing run" on a post-log artifact | medium | indicates an out-of-band change — find what ran without the CLI |
