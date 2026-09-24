# Step 18 — Compliance documentation & final review

The final step writes no code — because in MedTech, **evidence is a
deliverable**. The roadmap's Done criterion: *"a reviewer could audit our
data's journey end-to-end from the docs alone."*

## What we built

`docs/compliance/` — a reviewer-facing package:

| document | job |
|---|---|
| `README.md` | navigation: read order + which regulatory expectation maps to which artifact |
| `data-journey.md` | one image's path raw→model in 10 hops, each with its proof artifact |
| `audit-evidence.md` | *real outputs* from our own tools — chain INTACT, VERIFY: PASS, lineage trace, QC gate |
| `model-card.md` | baseline CNN: metrics, provenance, and **measured** limitations |
| `risk-register.md` | 10 risks rated L×I, mitigations mapped to implemented controls, accepted residuals stated |
| `sop-data-ingestion.md` | SOP-01: stage → ingest → labels → spot-check |
| `sop-dataset-release.md` | SOP-02: QC gate → publish → verify → card sign-off |
| `sop-annotation-review.md` | SOP-03: UI review workflow + the statistical rule |
| `sop-audit-review.md` | SOP-04: scheduled integrity review + escalation table |

Plus README polish: badges now honest (18/18 steps, 274 tests), a
Documentation section, and the `+cpu` install note that Step 16 taught us.

## Why this is a step, not a footnote

Everything built so far produces *capability*. Compliance asks a different
question: **can someone who wasn't here prove what happened?** That flips
the perspective from "does it work" to "can you demonstrate it worked."

Three design choices carry the step:

### 1. Evidence samples are real outputs, not descriptions

`audit-evidence.md` doesn't say "the audit log is verifiable" — it shows
`INTACT — chain intact` and the command that produced it, plus the tamper
demo result (`BROKEN at seq 2`). The difference matters: a claim can be
wrong; a pasted output with its command is *checkable*. A reviewer can
re-run every line.

### 2. The risk register maps mitigations to *implemented* controls

Each row names the control that exists — not the one that should. "API
speaks pseudonyms" cites the whole-response test that caught the Step 14
path leak. And the register lists **accepted risks** honestly (e.g.,
wholesale deletion of `artifacts/` is only recoverable from external
backup — the hash chain detects edits, not disappearance). A register that
claims everything is mitigated is a register nobody believes.

### 3. SOPs encode decisions, not just commands

Each SOP carries the non-obvious rules the code enforces: *fix the data,
never the gate*; *corrections are a new version, never an edit*; *report
the paired CI, never a winner claim* (Step 12's hard-won lesson); *a real
ID in the UI is a privacy incident — stop*. Procedures that only list
commands produce operators who can run them but can't think with them.

## The honest summary for the whole project

18 steps, one loop: raw data → manifest → curation → privacy → labels → QC
→ immutable releases → model → evaluation → active learning → audit → API
→ UI → packaging → cloud mapping → compliance. Every layer was verified
against the one below it, and the final package makes that verification
*auditable by someone else*.

## Verified

- Full suite: **274 passed, 1 skipped** (no code changes)
- Every command in the SOPs and evidence doc was run for real during Steps
  1–17 — nothing in the compliance package describes aspirational behavior
- README updated: badges, docs map, install note, all 18 checkboxes ✓
