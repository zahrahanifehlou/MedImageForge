# Step 7 — Annotation model & label store

```bash
python -m medimageforge load-labels        # CSV -> normalized label store
python -m medimageforge show-slice 049 14  # labels + mask + provenance, one call
```

This opens Phase 3.

---

## 1. What we built

Three new tables inside the manifest DB, plus two commands. The wide labels
CSV becomes a queryable label store with provenance.

## 2. Why it exists

**Labels are data too.** In `hemorrhage_diagnosis.csv` they are seven boolean
columns in a *wide* table. That shape cannot express:

- **who** made the annotation (a radiologist? a model? which version?)
- **when**, and from **which version** of which file
- a **new label type**, without `ALTER TABLE`

None of those are hypothetical. Step 12 will have a model propose labels for
the very same slices these radiologists annotated, and both sets must coexist.

---

## 3. Measure first — four findings that determined the schema

| Finding | Consequence for the design |
|---|---|
| **Mask ↔ label correspondence is exact**: 318 hemorrhage-positive slices, 318 masks, zero orphans in either direction | a mask reference belongs on the annotation record and can be trusted |
| **`No_Hemorrhage` == `NOT(any type)`** in all 2501 rows, 0 mismatches | it is **derived**, so we must *not* store it |
| **Fracture is independent**: 74 slices have a fracture and no hemorrhage | it is a different **category**, not a bleed type |
| **25 slices carry 2–3 types at once** | genuinely multi-label; a single "diagnosis" column would be wrong |

---

## 4. The schema — and why it is three tables, not one

```text
label_taxonomy                 slice_annotations                  slice_labels
─────────────────              ─────────────────────              ───────────────
code         (PK) ◄──────┐     id            (PK) ◄────────┐      annotation_id (FK)
display_name             │     patient_id                  └───── label_code    (FK)
category                 └──────────────────────────────────────  value  0|1
description                    slice_no
source_column                  mask_rel_path        ← per SLICE, not per label
                               source_file          ┐
                               source_sha256        │ provenance
                               annotator            │
                               annotated_at         ┘
                               UNIQUE(patient_id, slice_no, source_file, annotator)
```

**Why `mask_rel_path` is not in `slice_labels`.** The mask depends on the
*slice*, not on each individual label. Putting it in the label table would
repeat the same path six times per slice — a transitive dependency, and six
chances for the copies to drift apart.

**Why the taxonomy is its own table.** Adding an eighth label type becomes an
`INSERT`, not a schema migration. The wide CSV cannot do this: a new label
means a new column, which means changing every consumer.

**Why the identity includes `annotator` and `source_file`.** If identity were
just `(patient_id, slice_no)`, a second opinion would *overwrite* the first.
Instead both rows coexist and `get_slice_annotation` returns a **list** — one
entry per annotator. A test proves a disagreeing "model-v0" is stored
alongside the radiologists without either being lost.

**Why `No_Hemorrhage` has no taxonomy row.** We measured it as exactly
derivable, so it is computed on read:

```python
"no_hemorrhage": not hemorrhage,   # derived, never stored
```

Storing a value that is computable from others creates two sources of truth
that can disagree. This is the same instinct as Step 5's "run log vs state".

---

## 5. Provenance: `source_sha256`, not just a filename

Every annotation records the **SHA-256 of the CSV it came from**, not merely
its name. `hemorrhage_diagnosis.csv` could be revised tomorrow; "which file"
would still be true while meaning something different. The hash pins the exact
version — the same discipline as Step 4's manifest and Step 5's
`source_sha256`.

The `annotator` value is `radiologist-consensus`, taken from the dataset's own
README: *each slice was annotated by two radiologists who reached consensus*.
Real provenance, not a placeholder like `unknown`.

---

## 6. The "Done" criterion — one call

```text
$ python -m medimageforge show-slice 049 14
=== Patient 049 slice 14 ===
Hemorrhage types: ['epidural']
Other findings:   ['fracture']
No hemorrhage (derived): False
Mask: 049/brain/14_mask.png
Provenance:
  source_file: hemorrhage_diagnosis.csv
  source_sha256: fb25edc8308411a2...
  annotator: radiologist-consensus
  annotated_at: 2026-09-15T13:20:49+00:00
All labels: {'epidural': 1, 'intraparenchymal': 0, ..., 'fracture': 1}
```

This cross-checks against earlier steps: slice `049/14` is the epidural
hemorrhage whose mask we **visually verified** in Step 3, and the mask path
points at the **true-binary PNG** produced in Step 5. Four steps agreeing on
one slice is the reward for having a source of truth.

---

## 7. A hidden coupling the tests exposed

`_curated_mask_paths` originally did:

```sql
FROM files f LEFT JOIN curation c ON c.rel_path = f.rel_path
```

Eleven tests failed with `no such table: curation`. On *my* machine it worked —
because I had already run `curate`. The label store had silently acquired a
**dependency on Step 5 having run**, which is wrong: labels are meaningful
after `ingest` alone.

Fixed by probing `sqlite_master` for the table and falling back to raw mask
paths. The result is a real capability, not just a passing test: the store
degrades gracefully, preferring the curated PNG when it exists.

**Lesson:** a pipeline stage that works only because you happened to run an
earlier stage has an undocumented dependency. Tests on a fresh database are
what surface it.

---

## 8. Verified

| Check | Result |
|---|---|
| Slice annotations | 2501 (one per brain slice) |
| Label assertions | 15006 = 2501 × 6 taxonomy terms |
| With mask | 318 — matches the masks on disk exactly |
| Hemorrhage-positive | 318 — matches the mask count |
| Distribution from the **store** | Epidural 173, Intraparenchymal 73, Intraventricular 24, Subarachnoid 18, Subdural 56, Fracture 195 — identical to what Step 2 read from the CSV |
| Multi-label slices | 25, reproduced from the store |
| Re-load | idempotent: still 2501 / 15006 rows |
| Corrected label | updates in place, no duplicate row |
| Two annotators | coexist and stay distinguishable |
| `pytest` | **69 passed** |

## 9. Concepts learned

- **Schema before storage** — seven boolean columns become a taxonomy plus
  assertions, so labels gain identity, provenance, and extensibility.
- **Normalization has a purpose** — the mask lives with the slice because that
  is what it depends on; splitting the taxonomy makes new labels an INSERT.
- **Derive, don't duplicate** — `No_Hemorrhage` is computed, never stored.
- **Provenance pins versions, not names** — a file hash, not a filename.
- **Identity determines what can coexist** — including `annotator` in the key
  is what will let Step 12's model disagree with a radiologist on record.
