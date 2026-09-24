# SOP-03 — Annotation & active-learning review

| | |
|---|---|
| **Purpose** | Review model-suggested or model-flagged cases and feed confirmed labels back through a new dataset release. |
| **Scope** | UI review workflow; `active-learning` experiment loop; `load-labels` re-run. |
| **Roles** | Reviewer/annotator (clinical eye), data steward (release mechanics) |
| **Preconditions** | API running (`serve`), UI running (`ui`), current release verified |

## Procedure

1. **Open the review UI** — `python -m medimageforge ui` → Patients page.
   The UI shows **pseudonyms only**; the reviewer never sees a real ID.
2. **Inspect flagged/uncertain cases** — the active-learning report
   (`artifacts/runs/*/active_report.json`) lists patients ranked by model
   uncertainty; the UI's mask overlay shows the reference segmentation.
3. **Record decisions** in the label CSV per the labeling convention
   (subtype per slice, `No_Hemorrhage` for negatives). Never annotate from
   the model's overlay alone — the overlay is a *suggestion* to check, not
   the answer.
4. **Re-load labels** (SOP-01 step 3), **re-run QC**, **publish the next
   release** (SOP-02). Corrections are a new version, never an edit.

## The one statistical rule

When comparing selection strategies (uncertainty vs random), **report the
paired confidence interval**, not the mean delta. Step 12's 8-seed result
(CI `[−0.075, +0.063]`) showed the dataset cannot distinguish the two —
the honest output is "insufficient power," never a winner claim.

## Records produced

- updated label CSV + `load-labels` audit entry
- `datasets/vX.Y+1` with selection provenance in `metadata.json`
- active-learning comparison report (per-seed, per-arm metrics + CI)

## Failure handling

| Symptom | Action |
|---|---|
| reviewer disagrees with an existing label | record the disagreement; do not silently edit — resolve per labeling convention, then next release |
| UI shows a real patient ID | **stop — privacy incident**; file under risk R1; the boundary has been breached |
