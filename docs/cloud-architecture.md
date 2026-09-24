# MedImageForge — Cloud Architecture Mapping

**Status:** design document + Terraform sketch (`infra/`). No cloud account
required to read it; a cloud engineer could implement from this document.

**The one-sentence version:** every component we built is a *job*; the cloud
version is the same job on a different substrate. Local directories become
S3 buckets, the SQLite file becomes a managed database, shell invocations
become tasks in a state machine, and — the one real upgrade — privacy
conventions that existed only in code become IAM-enforced facts.

---

## 1. What we actually built (recap)

```
data/                    artifacts/                     datasets/
  Patients_CT/  (raw)      manifest.db  (SQLite)          v1.0, v1.1  (immutable)
  *.csv         (labels)   curated/     (validated zone)
.secrets/                  deid/        (pseudonym zone)
  pseudonym_salt           runs/, eval/ (models, reports)
                           audit.log    (hash chain)

CLI commands: ingest → curate → load-labels → qc → release
              train → evaluate → active-learning
              serve (FastAPI:8000)   ui (Streamlit:8501)
```

Everything above is described by **zones**: directories with a rule about
who may read or write them. The entire cloud mapping follows from that.

## 2. The substrate table

| Local component | What it does | AWS equivalent | Why |
|---|---|---|---|
| `data/Patients_CT/` | raw zone, never modified | **S3 `…-raw`** + versioning | S3 is the raw zone; versioning = "never modified" at the object level |
| `data/*.csv` | label/demographic inputs | same `…-raw` bucket, `inputs/` prefix | small structured files ride along with images |
| `artifacts/manifest.db` | SQLite source of truth | **RDS Postgres** | the manifest is now *concurrently* read (API) and written (pipeline) — a file can't do both |
| `artifacts/curated/` | validated zone | **S3 `…-curated`** | zone separation becomes bucket separation |
| `artifacts/deid/` | pseudonymized exports | **S3 `…-deid`** | the only bucket the API/UI roles may read |
| `datasets/` | immutable releases | **S3 `…-releases` + Object Lock** | Step 9's "published is immutable" becomes physical, not conventional |
| `.secrets/pseudonym_salt` | re-identification key | **Secrets Manager + KMS** | rotation, access logging, per-role grants |
| `artifacts/runs/`, `eval/` | model + eval outputs | **S3 `…-artifacts`** (+ MLflow/SageMaker Experiments later) | run records stay files; a tracker adds queryability |
| `artifacts/audit.log` | hash-chained log | `…-artifacts` **+ Object Lock** and **CloudTrail** | our chain detects tampering; Object Lock *prevents* it; CloudTrail audits the platform itself |
| CLI pipeline (`ingest…release`) | sequential data build | **Step Functions → ECS Fargate tasks** | retries, timeouts, execution history — the orchestration audit |
| `train` / `evaluate` | compute jobs | **AWS Batch** (GPU pool) | GPU on demand, scale-to-zero |
| `active-learning` loop | reviewer annotation queue | Batch + **SageMaker Ground Truth** (or the UI as the reviewer front-end) | "hide labels, reveal on demand" becomes a real labeling workflow |
| `serve` | FastAPI read API | **ECS Fargate service + internal ALB** | long-running, autoscaled |
| `ui` | Streamlit browser | **ECS Fargate service + ALB** | same pattern, public-facing behind auth |
| `Dockerfile` | the deployable unit | **ECR** (immutable tags) | Step 16's image *is* the artifact that ships |
| `ci.yml` | tests on every push | stays **GitHub Actions** + `docker push` + `terraform plan` | CI gains a deploy stage, not a rewrite |
| "someone runs qc" | manual invocation | **EventBridge Scheduler** (nightly cron) | the cron that remembers for you |
| `api.host = 127.0.0.1` | no-auth ⇒ no exposure | **Cognito/OIDC auth + ALB** | cloud removes the localhost excuse — auth becomes mandatory |

## 3. Target architecture

```
                        ┌────────────────────── VPC ──────────────────────┐
 Kaggle/scanner ──►     │  S3 raw ──► Step Functions ──► S3 curated       │
   upload               │            │  ingest          │                 │
                        │            │  curate           │ S3 deid        │
                        │            │  qc  ──FAIL──► ✋  │   ▲            │
                        │            │  release ──► S3   │   │ privacy    │
                        │            └──────┬───────────► releases (lock) │
                        │                   │           │   │            │
                        │   RDS Postgres ◄──┴───────────┘   │            │
                        │   (manifest)                      │            │
                        │        ▲                          │            │
                        │   ECS api ──────────────── reads deid+curated   │
                        │        ▲         (IAM: raw DENIED)              │
                        │   ECS ui ──── HTTP only (IAM: no S3 at all)     │
                        │        ▲                                        │
                        │   internal ALB ◄── ALB public ◄── Cognito auth  │
                        └─────────────────────────────────────────────────┘
   training:  Batch GPU job reads releases, writes runs/ → S3 artifacts
   audit:     CloudTrail (platform) + audit.log → Object-Locked artifacts bucket
   secrets:   Secrets Manager ◄── only pipeline_task role
```

## 4. The privacy boundary gets *stronger* — this is the real lesson

On localhost, three boundaries are **conventions enforced by code review**:

| Convention | Enforced by | Cloud equivalent | Enforced by |
|---|---|---|---|
| API returns pseudonyms only | `api.py` code + tests | `api_task` role has **no policy** for `…-raw` | IAM — the code *cannot* read real IDs' source zone |
| `.secrets/` never committed | `.gitignore` + discipline | Secrets Manager grant exists **only** on `pipeline_task` | IAM — rotation + access logs included |
| Releases are immutable | `release.py` refuses to overwrite | S3 Object Lock `GOVERNANCE`/`COMPLIANCE` | the storage layer — even an admin gets denied |
| Audit log append-only | hash chain detects edits | Object Lock prevents edits; CloudTrail logs platform actions | storage + platform audit |

**A bug in our code can no longer leak what the role cannot read.** Step 14's
real-ID-leak-in-a-path bug is the case study: in the cloud version, the buggy
response would have contained a `…-deid` path or failed with AccessDenied —
the raw zone isn't reachable from the API task *at all*.

## 5. What changes in the code

Deliberately little — this is why the zone/path abstraction in `configs/`
mattered from Step 1:

| change | size | notes |
|---|---|---|
| `manifest.py` SQLite → Postgres | medium | the SQL is plain; port dialect + replace `connect()` with a pooled DSN from env |
| file access → object access | medium | `Path.read_bytes()` sites get a `Storage` protocol with `LocalStorage` (tests/dev) and `S3Storage` implementations — the "one real adapter" the roadmap mentions |
| config paths → URIs | small | `data_dir: s3://mif-raw` — `config.py` already centralizes resolution |
| `serve --host` | trivial | `127.0.0.1` → `0.0.0.0` behind the ALB; auth middleware added |
| everything else | none | curation, QC, release logic, metrics, the model — all substrate-independent pure Python |

## 6. What's deliberately NOT in the sketch

- **Networking details** (VPC/subnets/security groups) — standard two-tier
  shape; `compute.tf` marks where subnet IDs go.
- **DICOM.** Real imaging arrives as DICOM, not JPG — ingestion would front
  HealthImaging or a DICOM-to-store gateway before `…-raw`.
- **AuthN/Z** — Cognito or OIDC; the sketch marks the boundary only.
- **Multi-account layout** — regulated deployments isolate PHI in its own
  account; out of scope here.
- **Cost model** — the honest headline: this dataset (~500 MB of artifacts)
  fits comfortably in free/near-free tiers; the real cost driver in
  production would be GPU training and storage at hospital scale.

## 7. Not-AWS equivalents (the mapping is conceptual)

| AWS | GCP | Azure |
|---|---|---|
| S3 + Object Lock | GCS + retention policy | Blob + immutable storage |
| RDS Postgres | Cloud SQL | Azure Database for PostgreSQL |
| ECS Fargate / Batch | Cloud Run / Batch | Container Apps / Batch |
| Step Functions | Workflows | Logic Apps / ADF |
| Secrets Manager + KMS | Secret Manager + Cloud KMS | Key Vault |
| CloudTrail | Cloud Audit Logs | Azure Monitor / Activity Log |

The zones, roles, and state machine carry over unchanged — only the
substrate names differ.
