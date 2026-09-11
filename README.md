# MedImageForge

## End-to-End Medical Imaging Data Platform for Continuous AI Model Improvement

MedImageForge is a platform for collecting, organizing, annotating, validating, versioning, and delivering **medical imaging datasets** for machine-learning development.

It is designed around the needs of regulated MedTech environments, where teams must know:

- Where each image came from
- How the image was processed
- Who changed or annotated it
- Which dataset version was used for a model
- Whether the data passed quality checks
- How access to sensitive data is controlled
- How datasets and workflows can be audited and reproduced

> **Important:** HIPAA/FDA compliance is not achieved by software alone. MedImageForge is designed to support compliant processes and controls, but actual compliance depends on the complete organization, infrastructure, procedures, validation, contracts, and regulatory requirements.

---

# This Is a Learning Project

We build this project **step by step**. The goal is not only a working system — it is to understand **why** each component exists, **what** it does, **how** it works, and **how the components connect**.

**How we work:**

1. Each step is small and has a clear goal.
2. For every step we answer four questions before coding:
   - **What** are we building?
   - **Why** does it exist? (What breaks without it?)
   - **How** does it work internally?
   - **How** does it connect to what we already built?
3. A step is only "done" when it runs, is verified, and is understood.
4. We never implement future steps early. If a later need appears, we note it and continue.

**Build order — local first, cloud later.** The full vision of this platform involves cloud infrastructure (AWS/Azure). We deliberately build everything **locally** first, because you cannot understand a cloud ingestion pipeline before you can load and inspect a single image. In Phase 6 we map every local component to its cloud equivalent — by then each component's job will be obvious, so the mapping will make sense.

---

# The Dataset We Use

All steps run on a real, public, anonymized dataset already present in `data/`:

**Computed Tomography Images for Intracranial Hemorrhage Detection and Segmentation**
(Hssayeni et al., PhysioNet — see `data/README.txt` and `data/LICENSE.txt`)

| Property | Value |
|---|---|
| Patients | 82 (`data/Patients_CT/049` … `130`) |
| Slices per patient | ~30 (5 mm slice thickness) |
| Image format | 650×650 grayscale JPG, two window settings per slice |
| Windows | `brain/` (soft tissue — used to see hemorrhage) and `bone/` (bone/fractures) |
| Slice labels | `data/hemorrhage_diagnosis.csv` — per slice: Intraventricular, Intraparenchymal, Subarachnoid, Epidural, Subdural, No_Hemorrhage, Fracture_Yes_No |
| Patient metadata | `data/patient_demographics.csv` — age, gender, patient-level conditions |
| Segmentation masks | `*_HGE_Seg.jpg` files inside each `brain/` folder — white region = hemorrhage |
| Integrity | `data/SHA256SUMS.txt` — official checksums for every file |

Why this dataset is ideal for learning:

- It is **real medical data**, already anonymized and published for research.
- It has **two "modalities"** (brain/bone windows), per-slice **labels**, and **segmentation masks** — enough to exercise every part of the platform.
- It is **small enough** to process on a laptop, but messy enough (JPGs instead of DICOM, inconsistent slice counts) to teach real-world curation lessons.

> The original `data/split_data.py` is the dataset author's reference script. It uses APIs that no longer exist (`scipy.misc.imread`). We will not rely on it — we will write our own, better pipeline and learn why in the process.

---

# The Roadmap

Six phases, eighteen steps. Checkboxes track our progress.

**Legend for every step:** `Build` = what we create · `Why` = the concept it teaches · `Done` = how we verify it.

---

## Phase 1 — Foundations & Knowing the Data

*Goal: a runnable project skeleton, and a real understanding of the data we will spend the whole project on.*

- [ ] **Step 1 — Project skeleton**
  - **Build:** Python package under `src/medimageforge/`, virtualenv, `requirements.txt`, `pyproject.toml` (editable install), YAML config loader, logging setup, first CLI command (`python -m medimageforge info`), first test.
  - **Why:** Every later component imports this foundation. You learn *why* code lives in an installable package (not loose scripts), *why* configuration lives outside code, and *why* we use structured logging instead of `print`.
  - **Done:** `python -m medimageforge info` prints the resolved config; `pytest` passes.

- [ ] **Step 2 — Dataset explorer**
  - **Build:** A script/command that scans `data/`, parses both CSVs, verifies files against `SHA256SUMS.txt`, and prints a report: patients, slice counts, window folders, label distribution, missing/extra files.
  - **Why:** You cannot build a pipeline for data you have not measured. Checksums teach **data integrity** — the first job of any ingestion system.
  - **Done:** Report matches reality (82 patients, ~2.5k slices/window); checksum verification passes or explicitly reports mismatches.

- [ ] **Step 3 — Image inspection**
  - **Build:** Load slices, display brain vs bone window side by side, overlay a `_HGE_Seg` mask on its slice, inspect pixel statistics.
  - **Why:** Understand CT windowing (same scan, different contrast for different tissue), what a segmentation mask is, and what a model will eventually see.
  - **Done:** You can visually confirm a mask aligns with a hemorrhage region.

## Phase 2 — Ingestion, Curation & Privacy

*Goal: raw files on disk become a trusted, queryable, de-identified dataset.*

- [ ] **Step 4 — Ingestion pipeline & manifest**
  - **Build:** An ingest command that registers every file into a **manifest** (SQLite DB): patient, slice, window, path, SHA-256, size, status. Re-runnable (idempotent).
  - **Why:** A manifest is the platform's source of truth — *you cannot manage what you cannot list*. SQLite teaches that a database beats a CSV once you need queries and updates.
  - **Done:** Manifest row count matches the explorer's count; re-running ingest changes nothing.

- [ ] **Step 5 — Curation pipeline**
  - **Build:** Validation (readable image? expected dimensions? has a label row?), duplicate detection via hashes, and normalized copies written to a separate `curated/` zone, plus a curation report.
  - **Why:** Learn **zone separation** (raw vs curated — never edit raw data in place) and **idempotent, resumable** pipelines.
  - **Done:** Curated zone contains only validated files; the report lists every rejection with a reason.

- [ ] **Step 6 — Privacy & de-identification gate**
  - **Build:** A check that verifies no PHI exists in our files (already anonymized JPGs — we prove it), plus pseudonymization of patient IDs in all *working* artifacts (e.g., `049` → `PAT-a3f9…`).
  - **Why:** Learn the difference between **anonymization** and **pseudonymization**, and why privacy is a *gate* that must pass before anything downstream runs.
  - **Done:** Gate produces a pass/fail privacy report; working artifacts contain no real patient numbers.

## Phase 3 — Annotations, Quality & Versioning

*Goal: labels become first-class data, datasets become reproducible releases.*

- [ ] **Step 7 — Annotation model & label store**
  - **Build:** A clean annotation schema (per-slice multi-label classification + segmentation mask reference) loaded into the manifest DB, with **provenance** (which source CSV, whose annotation).
  - **Why:** Labels are data too. Schema-before-storage is the lesson: five boolean columns in a CSV become a well-defined label taxonomy.
  - **Done:** For any slice we can query "labels + mask path + provenance" in one call.

- [ ] **Step 8 — Automated quality gates**
  - **Build:** A QC suite: image/label count mismatches, orphan files, mask-without-hemorrhage-flag inconsistencies, near-duplicate slices across different patients (leakage!), demographic outliers. Pass/fail quality report.
  - **Why:** A dataset must **earn** its way to training. Quality gates are the difference between a data lake and a data swamp.
  - **Done:** Report catches at least one real inconsistency in the raw data (this dataset has some — e.g., labels referencing missing slices).

- [ ] **Step 9 — Dataset versioning**
  - **Build:** Patient-level train/validation/test split, then an immutable snapshot `datasets/v1.0/` containing the split, an index of files, and a **dataset card** (what's inside, counts, known issues).
  - **Why:** Reproducibility — *"which data trained this model?"* must always be answerable. Splitting **by patient** (not by slice) is the single most important anti-leakage rule in medical imaging.
  - **Done:** `v1.0` is reproducible from the manifest; no patient appears in two splits.

## Phase 4 — ML Integration & the Improvement Loop

*Goal: versioned data flows into a model; model errors flow back as better data.*

- [ ] **Step 10 — Baseline model**
  - **Build:** PyTorch slice-level classifier (hemorrhage vs no hemorrhage) trained on `v1.0`. Each run records: dataset version, code version, config, metrics.
  - **Why:** Learn the Dataset → DataLoader → model → metrics chain, and why experiment records must reference a *dataset version*, not a folder.
  - **Done:** Model trains and beats a trivial baseline; run record is complete.

- [ ] **Step 11 — Evaluation & error analysis**
  - **Build:** Per-class metrics, confusion matrix, hardest slices ranked, slice-level → patient-level aggregation.
  - **Why:** Aggregate metrics lie. 82 patients means a single bad patient can swing accuracy; patient-level metrics are the honest number.
  - **Done:** We can name the specific slices/patients the model fails on.

- [ ] **Step 12 — Active learning loop**
  - **Build:** Score cases by model uncertainty → queue top-N "hard" cases → simulate a human reviewer (we use the existing labels as the "reviewer") → release `datasets/v1.1` → retrain → compare.
  - **Why:** This closes the platform's central loop: **Data → Model → Errors → Better data → Better model.**
  - **Done:** Documented before/after comparison on the same test split.

## Phase 5 — Platform Services

*Goal: the pipelines become a platform other programs can talk to.*

- [ ] **Step 13 — Lineage & audit log**
  - **Build:** Every pipeline run appends an audit record (who/what/when/inputs/outputs). Given any artifact, we can query its full ancestry back to raw ingestion.
  - **Why:** Traceability is the core regulated-MedTech requirement — and the best debugging tool you will ever build.
  - **Done:** Pick a file in `v1.1` and print its complete history.

- [ ] **Step 14 — API service**
  - **Build:** FastAPI service exposing patients, slices, labels, dataset versions, and QC status.
  - **Why:** A service boundary forces clean data access patterns and is the seam where access control will later live.
  - **Done:** Query the manifest over HTTP; the UI (Step 15) uses only this API.

- [ ] **Step 15 — Dataset browser UI**
  - **Build:** A minimal viewer (Streamlit or FiftyOne) to browse patients, windows, overlays, and labels through the API.
  - **Why:** Humans must be able to *see* the data — no platform survives without a viewer.
  - **Done:** Browse any patient; visually verify labels and masks.

## Phase 6 — Production-Readiness & Cloud Mapping

*Goal: the learning prototype becomes a defensible, documented system.*

- [ ] **Step 16 — Tests, CI & packaging**
  - **Build:** Fuller pytest suite, GitHub Actions CI, Dockerfile.
  - **Why:** A pipeline nobody can re-run or deploy is a script, not a platform.
  - **Done:** CI runs green on a clean checkout; Docker image runs the CLI.

- [ ] **Step 17 — Cloud architecture mapping**
  - **Build:** A documented mapping of every local component to its cloud equivalent — local dirs → S3 zones, ingest script → Lambda/S3 events, SQLite → RDS/DynamoDB, local training → SageMaker, our gates → IAM/RBAC policies — plus an IaC sketch. Optionally, one *real* cloud adapter (e.g., ingest from S3).
  - **Why:** Now that you know what each component *does*, the cloud version is "the same job, different substrate" — which is the correct mental model.
  - **Done:** An architecture document a cloud engineer could implement from.

- [ ] **Step 18 — Compliance documentation & final review**
  - **Build:** SOP templates, risk register, audit-trail evidence samples, dataset/model cards, final README polish.
  - **Why:** In MedTech, *evidence* is a deliverable. This step produces it.
  - **Done:** A reviewer could audit our data's journey end-to-end from the docs alone.

---

# Core Capabilities (the "why" behind the roadmap)

## 1. Secure ingestion

Continuously receive large volumes of imaging data with validation, integrity checks, metadata handling, logging, and controlled access. → *Steps 2, 4, 6*

## 2. Data curation

Raw data is not ML-ready. Organize, detect invalid files and duplicates, validate metadata, tag, filter, and track preprocessing. → *Step 5*

## 3. Annotation & AI-assisted annotation

Models need labels: classes, boxes, and 3D/2D segmentation masks. An existing model can *suggest* labels that a human accepts/corrects/rejects — reducing effort while keeping human oversight. → *Steps 7, 12, 15* (integrations like CVAT/FiftyOne/MONAI Label are optional extensions)

## 4. Active learning

Don't annotate randomly — send the model's *uncertain* cases to annotators. Feedback loop: **Data → Annotation → Model → Errors → Better data.** → *Steps 11–12*

## 5. Data lineage

*"Where did this image come from and what happened to it?"* — original → de-identified → resampled → QC'd → annotated → dataset v1.4 → model v2.1. → *Step 13*

## 6. Dataset versioning

Datasets change: v1.0 → v1.1 → v2.0. Always know exactly which version trained or validated a model. → *Step 9* (DVC/object storage are optional extensions)

## 7. Quality control

Automated checks for missing/corrupt files, invalid metadata, duplicates, wrong dimensions, label inconsistencies, and train/test leakage — enforced as gates before release. → *Step 8*

## 8. Governance & access control

RBAC, least-privilege, encryption, audit logging, retention, controlled dataset access. Example roles: Data Engineer → ingestion; Annotator → annotation; ML Engineer → training sets; QA → audit. → *Steps 6, 13, 14, 17*

## 9. Regulatory support

De-identification, audit trails, SOPs, validation protocols, risk assessment, change control, traceability, documentation — evidence that data was handled under control. → *Steps 6, 13, 18*

---

# Key Technologies

| Area | We actually use | Vision-level (mapped in Phase 6) |
|---|---|---|
| Language | Python 3.10 | Python (+ C++ only if a profiled hotspot demands it) |
| Medical imaging | Pillow, NumPy | SimpleITK, ITK, OpenCV, MONAI |
| Labels/masks | pandas, our schema | CVAT, FiftyOne, MONAI Label |
| ML | PyTorch | PyTorch, TensorFlow |
| Data platform | SQLite manifest, filesystem zones | DVC, S3, Glue, RDS/DynamoDB |
| Services | FastAPI (Step 14) | API Gateway, Lambda, SageMaker |
| Ops | pytest, GitHub Actions, Docker | CI/CD, IaC, monitoring, SLOs |

---

# Architecture Concept

```text
             Medical Data Sources          (data/Patients_CT — our dataset)
                     |
                     v
              Secure Ingestion             Step 4: manifest + checksums
                     |
          +----------+-----------+
          |   Raw / Controlled   |         raw zone — never modified
          +----------+-----------+
                     |
                     v
             De-identification             Step 6: privacy gate
                     |
                     v
               Data Curation               Step 5: curated zone + report
                     |
          +----------+----------+
          |                     |
          v                     v
     Quality Control       Annotation      Steps 8 / 7
          |                     |
          +----------+----------+
                     |
                     v
             Dataset Versioning            Step 9: datasets/v1.0, v1.1, ...
                     |
                     v
              AI / ML Platform             Steps 10–11
                     |
          +----------+----------+
          |                     |
          v                     v
      Training             Evaluation
          |                     |
          +----------+----------+
                     |
                     v
             Active Learning               Step 12
                     |
                     v
            New / Hard Cases  --->  back to Annotation  --->  Improved Dataset
```

Services layer (Steps 13–15) wraps all of the above: audit log, API, browser UI.
Cloud layer (Step 17) maps each box to AWS/Azure.

---

# Design Principles

1. **Security by design** — built in, not bolted on.
2. **Privacy by design** — minimize, protect, pseudonymize working copies.
3. **Reproducibility** — an experiment = code version + dataset version + config + environment.
4. **Traceability** — always answerable: *who changed what, when, why, and what did it affect?*
5. **Automation** — repetitive manual work becomes tested, observable pipelines.
6. **Human oversight** — AI-suggested labels still need review.
7. **Separation of environments** — dev / test / validation / production are distinct.
8. **Never modify raw data** — raw is immutable; everything else is derived and regenerable.

---

# Project Structure

```text
MedImageForge/
│
├── README.md                  ← you are here (vision + roadmap)
├── requirements.txt           ← runtime dependencies (grows per step)
├── pyproject.toml             ← makes src/medimageforge installable (pip install -e .)
├── .gitignore
│
├── configs/
│   └── default.yaml           ← all paths/settings live here, not in code
│
├── data/                      ← CT-ICH dataset (gitignored — never commit data)
│   ├── Patients_CT/           ← 82 patients × (brain/ + bone/ windows)
│   ├── hemorrhage_diagnosis.csv
│   ├── patient_demographics.csv
│   ├── SHA256SUMS.txt
│   └── README.txt, LICENSE.txt, ct_ich.yml, split_data.py   (dataset originals)
│
├── docs/
│   └── steps/
│       └── step-01-*.md       ← per-step learning notes: what/why/how
│
├── src/
│   └── medimageforge/         ← the installable package (all logic lives here)
│       ├── __init__.py
│       ├── __main__.py        ← enables: python -m medimageforge
│       ├── cli.py             ← command-line entry point
│       ├── config.py          ← config loading + path resolution
│       └── logging_utils.py   ← one logging setup for the whole platform
│
├── scripts/                   ← thin wrappers that call into the package
├── notebooks/                 ← exploration (Step 3)
└── tests/                     ← pytest suite (grows per step)
```

---

# Getting Started

```powershell
# 1. Clone and enter
git clone <repo-url>; cd MedImageForge

# 2. Create and activate the virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies + the package itself (editable)
pip install -r requirements.txt
pip install -e .

# 4. Verify the foundation works
python -m medimageforge info
pytest
```

---

# Medical Data Safety

**Never commit real patient data to this repository.** `data/` is gitignored.

Our dataset is already anonymized and public — but we still treat `data/` as untrusted input and keep all *derived* artifacts pseudonymized, to practice the habits a real MedTech system requires.

Do not add: patient names, MRNs, dates of birth, addresses, contact details, direct identifiers, unapproved DICOM containing PHI, or any sensitive patient information.

---

# Expected Benefits

A completed MedImageForge demonstrates: reduced manual data work, higher dataset quality, reproducible ML experiments, tracked dataset changes, efficient annotation, automatic hard-case discovery, stronger security/auditability, and a documented path to regulated development.

# Success Criteria

- Securely ingest imaging data and prove file integrity
- Handle 2D slices and multi-window studies
- Pass a privacy gate and keep working artifacts pseudonymized
- Curate automatically with a rejection report
- Query labels + masks + provenance in one call
- Enforce quality gates before dataset release
- Produce immutable, patient-split dataset versions
- Train and evaluate a model on a named dataset version
- Feed model errors back into an improved dataset (v1.1)
- Trace any artifact's full lineage; expose the platform via API

---

# Compliance Notice

MedImageForge is designed to **support** medical-data security, privacy, quality, traceability, and regulated software-development processes. It must not be described as inherently "HIPAA-compliant" or "FDA-compliant." Actual compliance depends on architecture, cloud configuration, security controls, organizational policies, BAAs where applicable, validation activities, quality-management processes, intended use, product classification, and applicable law. A formal compliance assessment is required for any real production implementation.

---

# License

**Proprietary — Internal MedTech use only.**

Dataset license: see `data/LICENSE.txt` (PhysioNet — PhysioNet Credentialed Health Data License / dataset-specific terms).
