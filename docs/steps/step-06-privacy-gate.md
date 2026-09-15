# Step 6 — Privacy & de-identification gate

```bash
python -m medimageforge privacy      # exit code 0 = PASS, 1 = FAIL
```

This step closes Phase 2.

---

## 1. The two words everyone confuses

| | Reversible? | Key exists? | Still personal data? |
|---|---|---|---|
| **Anonymization** | no | no | no — outside privacy law |
| **Pseudonymization** | yes, by the key holder | yes | **yes** — still regulated |

- Our **images** are anonymized at source: the dataset authors exported them
  from DICOM to JPEG, discarding the headers. We *verify* this rather than
  trust it.
- Our **patient numbers** we **pseudonymize**: `049` → `PAT-98c32af848a5`.

Why not anonymize the IDs too? Because a platform that cannot answer "which
patient was this?" for an authorized investigator is useless clinically. We
deliberately keep a re-identification path — and then protect it.

## 2. Why it is a *gate*, not a report

`cmd_privacy` returns a non-zero exit code on failure. That is the whole
point: CI, a release script, or Step 9 can check the exit status and refuse
to proceed. A privacy check nobody is forced to honour is decoration.

---

## 3. The most important line of code

```python
digest = hmac.new(salt, patient_id.encode(), hashlib.sha256).hexdigest()
return f"PAT-{digest[:12]}"
```

**A plain hash is not pseudonymization for a small identifier space.**
Patient IDs here are 3-digit numbers — 1000 candidates. `sha256("049")` is a
fixed public value, so an attacker builds a rainbow table in microseconds.
`tests/test_privacy.py` performs that exact attack:

```python
rainbow = {sha256(f"{i:03d}".encode()).hexdigest(): f"{i:03d}" for i in range(1000)}
rainbow[sha256("049".encode()).hexdigest()] == "049"   # identity recovered
```

The same test then shows the attack **fails** against our HMAC output. The
secret salt is what makes the token opaque — the algorithm alone does not.

### Where the key lives

| | |
|---|---|
| Precedence | `MEDIMAGEFORGE_PSEUDONYM_SALT` env var, else `.secrets/pseudonym_salt` |
| Permissions | file created `0600` (owner-only), verified by a test |
| Git | `.secrets/` is gitignored — confirmed with `git check-ignore` |
| Risk | **lose the salt and every pseudonym becomes permanently unlinkable** — pseudonymization silently degrades into anonymization |

Pseudonyms are **deterministic**: the same patient always yields the same
token, so pseudonymized artifacts still join to one another. That is a
requirement, not a convenience.

---

## 4. The zone trust model (a decision we made explicitly)

The README says "pseudonymize all working artifacts". We chose the
**controlled-zone** interpretation:

```text
┌─ CONTROLLED ZONE — access-controlled, real IDs permitted ───────┐
│  data/                     raw dataset (049/brain/14.jpg)       │
│  artifacts/manifest.db     manifest + `patients` mapping table  │
│  artifacts/curated/        curated zone (049/brain/14.png)      │
└─────────────────────────────────────────────────────────────────┘
                              │  pseudonymization boundary
                              ▼
┌─ MAY LEAVE THE ZONE — pseudonyms only ─────────────────────────┐
│  artifacts/deid/*.csv      PAT-98c32af848a5, no real numbers    │
│  (Step 9 releases, Step 14 API responses will follow this rule) │
└─────────────────────────────────────────────────────────────────┘
```

The `patients` table (real ID ↔ pseudonym) lives **inside** the controlled
zone. That table *is* the re-identification path — which is precisely what
makes this pseudonymization and not anonymization. Storing it anywhere that
travels with the exports would defeat the entire exercise.

---

## 5. The five checks, and what each defends against

| Check | Defends against | Result |
|---|---|---|
| `image-metadata` | EXIF / JPEG comments carrying `PatientName`, `StudyDate` — the classic DICOM-header leak | PASS — 5319 images, **zero** metadata of any kind |
| `identifier-columns` | a column named `name`/`mrn`/`dob`/`address`… ever being ingested | PASS |
| `free-text-identifiers` | emails, phones, SSNs, MRNs, dates, `Dr. Smith` hiding in prose | PASS |
| `age-over-89` | HIPAA Safe Harbor: extreme ages identify people in small cohorts | PASS — max age is 72 |
| `no-real-ids-in-exports` | the export still permitting re-identification | PASS |

Notable data findings:

- The only free-text note reads *"CT scan was after about two weeks of the
  accident"* — a **relative interval**, which is Safe-Harbor-compatible. An
  actual date would not be. A test pins this distinction.
- `Condition on file` holds 13 clinical strings only (`Subdural HGE`, …).
- Minimum age is 0.0027 years (~1 day old). Real, and permitted.

---

## 6. The false positive that rewrote a check

The first version of `no-real-ids-in-exports` searched the exported text for
the digits of every real patient ID. **The gate failed:**

```text
[FAIL] no-real-ids-in-exports
     - demographics_pseudonymized.csv: real-patient-id-present: '49'
```

Investigation: row 33 of the export is a **49-year-old patient**. Patient
`049` also exists. Patient IDs run 49–130; ages run 0–72. **The ranges
overlap, so a text search cannot possibly distinguish an age from an ID.**

The invariant we actually need is not *"these digits appear nowhere"* but:

> **no column reproduces the patient identity for its own row.**

A leak *aligns* with the real ID row by row; a coincidence does not. So the
check became **row-aligned**, flagging a column only when it matches far more
often than chance:

```python
for value, pid in zip(frame[col], real_ids):   # compared PER ROW
    if same_id(value, pid):
        aligned += 1
if aligned / total > threshold:                # 1.0 for a real leak
    flag(col)
```

Tests pin both sides so the fix cannot silently become permissive:

- `test_coincidental_number_is_not_a_leak` — age 49 on another patient: clean
- `test_column_reproducing_the_patient_id_is_a_leak` — fraction 1.0: flagged

**Lesson:** when a gate fails, first establish whether the *data* or the
*gate* is wrong. Loosening a check to make it green is how privacy controls
die. Here the check was genuinely wrong — but only investigation could show
that, and the fix made it *stricter* in the cases that matter.

### A second bug the tests caught

The rewritten check derived candidate ID forms from the CSV's bare integers
(`49`), so a column containing the **zero-padded folder form** (`"049"`) — a
genuine leak — was missed. `test_zero_padded_id_is_also_caught` failed;
comparing numerically fixed it. Two bugs in one step, both found by tests
written against data the real dataset never produces.

---

## 7. Known limitations (residual risk, stated honestly)

- **Burned-in pixel text is not detected.** Scanner overlays can render a
  patient's name into the image itself; catching that needs OCR, which we
  have not added. Mitigation: these images were exported from video frames
  and visual inspection in Step 3 showed no overlays. This remains a
  documented residual risk, not a solved problem.
- **Quasi-identifiers are not eliminated.** Age + gender + diagnosis could
  re-identify someone in a small cohort. Safe Harbor permits these fields; a
  full k-anonymity analysis is out of scope.
- **Regex detectors are heuristics.** They prove we looked; they cannot prove
  absence.

---

## 8. Verified

```text
[PASS] image-metadata            5319 images, no EXIF/comments
[PASS] identifier-columns
[PASS] free-text-identifiers
[PASS] age-over-89
[PASS] no-real-ids-in-exports    row-aligned, 2 exports
GATE: PASS   (exit code 0)
```

- 82 patients pseudonymized, 82 unique tokens, no collisions
- `049 → PAT-98c32af848a5` stable across separate runs
- `.secrets/pseudonym_salt` mode `600`, confirmed gitignored
- exports contain a `patient` column of pseudonyms and **no** real ID column
- `pytest` → 55 passed
