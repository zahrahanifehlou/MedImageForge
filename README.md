# MedImageForge

<p align="center">
  <img src="docs/images/medimageforge-logo.png" alt="MedImageForge Logo" width="520"/>
</p>

<p align="center">
  <strong>End-to-End Medical Imaging Data Platform<br>for Continuous AI Model Improvement</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Phase-1%20%E2%80%93%203-green?style=flat-square" alt="Phases 1-3 complete"/>
  <img src="https://img.shields.io/badge/Phase-4%20%E2%80%93%206-lightgrey?style=flat-square" alt="Phases 4-6 pending"/>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square" alt="Python"/>
  <img src="https://img.shields.io/badge/License-Proprietary-red?style=flat-square" alt="License"/>
</p>

---

**MedImageForge** is a platform for collecting, organizing, annotating, validating, versioning, and delivering **medical imaging datasets** for machine-learning development.

It is built for regulated MedTech environments, where teams must always know:

| Question | Why it matters |
|----------|----------------|
| Where each image came from | Provenance & chain of custody |
| How the image was processed | Reproducibility |
| Who changed or annotated it | Accountability |
| Which dataset version trained a model | Experiment tracking |
| Whether the data passed quality checks | Trustworthiness |
| How access to sensitive data is controlled | Privacy & security |
| How datasets and workflows can be audited | Regulatory readiness |

---

## Local First Philosophy

> **Build order — local first, cloud later.**

The full vision includes cloud infrastructure (AWS / Azure).  
We deliberately build **everything locally first**, because you cannot understand a cloud ingestion pipeline until you can load and inspect a single image.

In **Phase 6** we map every local component to its cloud equivalent. By then each component’s job is obvious, so the mapping becomes straightforward.

---

## Dataset

All steps run on a real, public, anonymized dataset already present in `data/`:

**Computed Tomography Images for Intracranial Hemorrhage Detection and Segmentation**  
*(Hssayeni et al., PhysioNet — see `data/README.txt` and `data/LICENSE.txt`)*

| Property              | Value |
|-----------------------|-------|
| Patients              | 82 (`data/Patients_CT/049` … `130`) |
| Slices per patient    | ~30 (5 mm slice thickness) |
| Image format          | 650×650 grayscale JPG, two window settings per slice |
| Windows               | `brain/` (soft tissue — hemorrhage) and `bone/` (bone / fractures) |
| Slice labels          | `data/hemorrhage_diagnosis.csv` — multi-label per slice |
| Patient metadata      | `data/patient_demographics.csv` — age, gender, conditions |
| Segmentation masks    | `*_HGE_Seg.jpg` inside each `brain/` folder |
| Integrity             | `data/SHA256SUMS.txt` — official checksums |

### Why this dataset is ideal for learning

- Two “modalities” (brain / bone windows) + per-slice labels + segmentation masks → exercises every platform component
- Small enough to process on a laptop, yet messy enough (JPGs instead of DICOM, inconsistent slice counts) to teach real-world curation

> The original `data/split_data.py` is the dataset author’s reference script. It uses deprecated APIs (`scipy.misc.imread`). We will not rely on it — we write our own, better pipeline and learn *why* in the process.

---

## The Roadmap

Six phases • Eighteen steps. Checkboxes track progress.

**Legend for every step**  
`Build` = what we create · `Why` = the concept it teaches · `Done` = how we verify it

---

### Phase 1 — Foundations & Knowing the Data

*Goal: a runnable project skeleton and a real understanding of the data.*

- [x] **Step 1 — Project skeleton**  
  **Build:** Python package under `src/medimageforge/`, virtualenv, `requirements.txt`, `pyproject.toml`, YAML config, logging, first CLI (`python -m medimageforge info`), first test.  
  **Why:** Every later component imports this foundation. Learn why code lives in an installable package, why configuration lives outside code, and why we use structured logging.  
  **Done:** `python -m medimageforge info` prints resolved config; `pytest` passes.

- [x] **Step 2 — Dataset explorer**  
  **Build:** Script that scans `data/`, parses both CSVs, verifies files against `SHA256SUMS.txt`, and prints a full report.  
  **Why:** You cannot build a pipeline for data you have not measured. Checksums teach **data integrity**.  
  **Done:** Report matches reality (82 patients, ~2.5k slices/window); checksum verification passes or reports mismatches.

- [x] **Step 3 — Image inspection**  
  **Build:** Load slices, display brain vs bone side-by-side, overlay `_HGE_Seg` mask, inspect pixel statistics.  
  **Why:** Understand CT windowing, what a segmentation mask is, and what a model will eventually see.  
  **Done:** Visually confirm a mask aligns with a hemorrhage region.

---

### Phase 2 — Ingestion, Curation & Privacy

*Goal: raw files become a trusted, queryable, de-identified dataset.*

- [x] **Step 4 — Ingestion pipeline & manifest**  
  **Build:** Ingest command that registers every file into a **manifest** (SQLite): patient, slice, window, path, SHA-256, size, status. Idempotent.  
  **Why:** A manifest is the platform’s source of truth. SQLite beats a CSV once you need queries and updates.  
  **Done:** Manifest row count matches explorer; re-running ingest changes nothing.

- [x] **Step 5 — Curation pipeline**  
  **Build:** Validation, duplicate detection, normalized copies written to a separate `curated/` zone + curation report.  
  **Why:** **Zone separation** (raw vs curated — never edit raw data in place) and **idempotent, resumable** pipelines.  
  **Done:** Curated zone contains only validated files; report lists every rejection with reason.

- [x] **Step 6 — Privacy & de-identification gate**  
  **Build:** Check that no PHI exists + pseudonymization of patient IDs in working artifacts (`049` → `PAT-a3f9…`).  
  **Why:** Difference between **anonymization** and **pseudonymization**. Privacy is a *gate* that must pass before anything downstream runs.  
  **Done:** Gate produces pass/fail privacy report; working artifacts contain no real patient numbers.

---

### Phase 3 — Annotations, Quality & Versioning

*Goal: labels become first-class data; datasets become reproducible releases.*

- [x] **Step 7 — Annotation model & label store**  
  **Build:** Clean annotation schema (per-slice multi-label + segmentation mask reference) loaded into the manifest DB with **provenance**.  
  **Why:** Labels are data too. Schema-before-storage is the lesson.  
  **Done:** For any slice we can query “labels + mask path + provenance” in one call.

- [x] **Step 8 — Automated quality gates**  
  **Build:** QC suite for mismatches, orphans, mask/label inconsistencies, near-duplicates (leakage!), demographic outliers.  
  **Why:** A dataset must **earn** its way to training. Quality gates turn a data lake into a trusted asset.  
  **Done:** Report catches at least one real inconsistency in the raw data.

- [x] **Step 9 — Dataset versioning**  
  **Build:** Patient-level train/val/test split → immutable snapshot `datasets/v1.0/` + dataset card.  
  **Why:** Reproducibility — “which data trained this model?” must always be answerable. Patient-level split is the #1 anti-leakage rule.  
  **Done:** `v1.0` is reproducible from the manifest; no patient appears in two splits.

---

### Phase 4 — ML Integration & the Improvement Loop

*Goal: versioned data flows into a model; model errors flow back as better data.*

- [x] **Step 10 — Baseline model**  
  **Build:** PyTorch slice-level classifier trained on `v1.0`. Each run records dataset version, code version, config, metrics.  
  **Why:** Dataset → DataLoader → model → metrics chain. Experiment records must reference a *dataset version*.  
  **Done:** Model trains and beats a trivial baseline; run record is complete.

- [x] **Step 11 — Evaluation & error analysis**  
  **Build:** Per-class metrics, confusion matrix, hardest slices, patient-level aggregation.  
  **Why:** Aggregate metrics lie. Patient-level metrics are the honest number.  
  **Done:** We can name the specific slices/patients the model fails on.

- [x] **Step 12 — Active learning loop**  
  **Build:** Score by uncertainty → queue hard cases → simulate reviewer → release `datasets/v1.1` → retrain → compare.  
  **Why:** Closes the central loop: **Data → Model → Errors → Better data → Better model**.  
  **Done:** Documented before/after comparison on the same test split.

---

### Phase 5 — Platform Services

*Goal: the pipelines become a platform other programs can talk to.*

- [ ] **Step 13 — Lineage & audit log**  
  **Build:** Every pipeline run appends an audit record. Query full ancestry of any artifact.  
  **Why:** Traceability is the core regulated-MedTech requirement — and the best debugging tool.  
  **Done:** Pick a file in `v1.1` and print its complete history.

- [ ] **Step 14 — API service**  
  **Build:** FastAPI service exposing patients, slices, labels, dataset versions, QC status.  
  **Why:** A service boundary forces clean data access patterns and is the future home of access control.  
  **Done:** Query the manifest over HTTP; the UI uses only this API.

- [ ] **Step 15 — Dataset browser UI**  
  **Build:** Minimal viewer (Streamlit or FiftyOne) to browse patients, windows, overlays, and labels.  
  **Why:** Humans must be able to *see* the data.  
  **Done:** Browse any patient; visually verify labels and masks.

---

### Phase 6 — Production-Readiness & Cloud Mapping

*Goal: the learning prototype becomes a defensible, documented system.*

- [ ] **Step 16 — Tests, CI & packaging**  
  **Build:** Fuller pytest suite, GitHub Actions CI, Dockerfile.  
  **Why:** A pipeline nobody can re-run or deploy is a script, not a platform.  
  **Done:** CI runs green on a clean checkout; Docker image runs the CLI.

- [ ] **Step 17 — Cloud architecture mapping**  
  **Build:** Documented mapping of every local component to its cloud equivalent + IaC sketch. Optionally one real cloud adapter.  
  **Why:** Once you know what each component *does*, the cloud version is “the same job, different substrate”.  
  **Done:** An architecture document a cloud engineer could implement from.

- [ ] **Step 18 — Compliance documentation & final review**  
  **Build:** SOP templates, risk register, audit-trail evidence samples, dataset/model cards, final README polish.  
  **Why:** In MedTech, *evidence* is a deliverable.  
  **Done:** A reviewer could audit our data’s journey end-to-end from the docs alone.

---

## Core Capabilities

| Capability | Description | Steps |
|------------|-------------|-------|
| **1. Secure ingestion** | Continuously receive imaging data with validation, integrity checks, metadata, logging, and controlled access | 2, 4, 6 |
| **2. Data curation** | Organize, detect invalid/duplicate files, validate metadata, tag, filter, track preprocessing | 5 |
| **3. Annotation & AI-assisted annotation** | Classes, boxes, 2D/3D segmentation. Models can suggest labels for human review | 7, 12, 15 |
| **4. Active learning** | Send the model’s uncertain cases to annotators. Feedback loop: Data → Annotation → Model → Errors → Better data | 11–12 |
| **5. Data lineage** | “Where did this image come from and what happened to it?” Full chain from raw → model | 13 |
| **6. Dataset versioning** | Immutable snapshots (v1.0 → v1.1 → v2.0). Always know which version trained a model | 9 |
| **7. Quality control** | Automated checks for missing/corrupt files, bad metadata, duplicates, leakage — enforced as gates | 8 |
| **8. Governance & access control** | RBAC, least-privilege, encryption, audit logging, retention | 6, 13, 14, 17 |
| **9. Regulatory support** | De-identification, audit trails, SOPs, validation protocols, risk assessment, traceability | 6, 13, 18 |

---

## Key Technologies

| Area              | We actually use                          | Vision-level (Phase 6)                          |
|-------------------|------------------------------------------|-------------------------------------------------|
| Language          | Python 3.10                              | Python (+ C++ only if profiled hotspot)         |
| Medical imaging   | Pillow, NumPy                            | SimpleITK, ITK, OpenCV, MONAI                   |
| Labels / masks    | pandas, our schema                       | CVAT, FiftyOne, MONAI Label                     |
| ML                | PyTorch                                  | PyTorch, TensorFlow                             |
| Data platform     | SQLite manifest, filesystem zones        | DVC, S3, Glue, RDS / DynamoDB                   |
| Services          | FastAPI (Step 14)                        | API Gateway, Lambda, SageMaker                  |
| Ops               | pytest, GitHub Actions, Docker           | CI/CD, IaC, monitoring, SLOs                    |

---

## Architecture Concept

```text
Medical Data Sources (data/Patients_CT)
          │
          ▼
Secure Ingestion  (Step 4)  ──►  manifest + checksums
          │
          ▼
     ┌────┴────┐
     │ Raw zone │  ← never modified
     └────┬────┘
          ▼
De-identification Gate  (Step 6)
          │
          ▼
Data Curation  (Step 5)  ──►  curated/ zone + report
          │
     ┌────┴────┐
     ▼         ▼
Quality     Annotation
Control     (Steps 8 / 7)
     │         │
     └────┬────┘
          ▼
Dataset Versioning  (Step 9)  ──►  datasets/v1.0, v1.1, …
          │
          ▼
AI / ML Platform  (Steps 10–11)
          │
     ┌────┴────┐
     ▼         ▼
 Training   Evaluation
     │         │
     └────┬────┘
          ▼
Active Learning  (Step 12)
          │
          ▼
New / Hard Cases  ──►  back to Annotation  ──►  Improved Dataset

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
