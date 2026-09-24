# Step 17 — Cloud architecture mapping

The last Phase 6 step, and the one that asks no new code of the platform:
*"once you know what each component does, the cloud version is the same job,
different substrate."* Deliverable: an architecture document a cloud engineer
could implement from — plus a Terraform sketch.

## What we built

| artifact | purpose |
|---|---|
| `docs/cloud-architecture.md` | the main deliverable: full component mapping, target architecture diagram, privacy-boundary analysis, code-change inventory, AWS↔GCP↔Azure equivalence table |
| `infra/main.tf` | provider + variables + the mapping rule in a comment |
| `infra/storage.tf` | five S3 buckets = the five local zones; Object Lock on `releases` and `artifacts` |
| `infra/secrets.tf` | `.secrets/pseudonym_salt` → Secrets Manager + KMS |
| `infra/database.tf` | `manifest.db` (SQLite) → RDS Postgres |
| `infra/compute.tf` | ECR repo, ECS task defs for `serve`/`ui`, AWS Batch GPU pool for `train` |
| `infra/iam.tf` | the privacy boundary as IAM policy — API/UI roles can't reach raw |
| `infra/pipeline.tf` | CLI chain → Step Functions state machine + nightly EventBridge trigger |
| `infra/outputs.tf` | what a deploy would hand back (bucket names, ECR URL, DB endpoint) |

## The one idea that organizes everything

Every local path in `configs/default.yaml` is a **zone** — a directory plus a
rule about who may touch it. In the cloud, a zone is a bucket (or a database)
plus an IAM policy. `data/Patients_CT/` → `s3://mif-raw`. `datasets/` →
`s3://mif-releases`. That's the whole trick; the rest is per-component detail.

## Why the mapping is honest, not hand-wavy

Each row of the substrate table answers three questions: what the local thing
*does*, what replaces it, and *what breaks if you pick the naive equivalent*:

- **`manifest.db` → RDS Postgres**, not "SQLite on a mounted volume." The
  manifest is now read by the API *while* the pipeline writes it — a file on
  a filesystem can't be concurrently shared between services. The substrate
  change is forced by the architecture, not fashion.
- **`train` → AWS Batch GPU**, not a bigger Fargate task. Training is a
  *job* (queued, retried, exits); a service is *running* (restarted if it
  exits). Different substrate because they're different shapes of compute.
- **`audit.log` → Object Lock + CloudTrail.** Our hash chain *detects*
  tampering; Object Lock *prevents* it; CloudTrail audits the platform's own
  actions (who touched the bucket), which a self-written log never can.

## The finding worth carrying forward

Three of our privacy guarantees were **conventions enforced by code** —
the API speaks pseudonyms, the salt is gitignored, releases are immutable.
In the cloud each becomes a **fact enforced by infrastructure**: the API
task role has no policy for the raw bucket, the salt secret is granted only
to the pipeline role, Object Lock denies even admins.

Step 14's real-IDs-in-a-path bug is the case study: in the mapped
architecture, that buggy code path would have hit `AccessDenied` instead of
leaking `Patients_CT/049/...` — **a bug can't leak what the role can't
read.** "The UI uses only this API" (Step 15's contract, proven by tests)
similarly becomes literally true: the UI task role has *zero* S3 grants.

## What changed in the codebase

Nothing — deliberately. The only honest cost inventory is in the doc:
`manifest.py` needs a Postgres dialect port, file access needs a `Storage`
protocol (the roadmap's "one real adapter" slot), config paths become URIs.
Everything else — curation logic, QC gates, release math, metrics, the
model — is substrate-independent pure Python. **That was the payoff of
keeping zones and config centralized since Step 1.**

## Verified

- `infra/*.tf` files parse as HCL (structure checked by eye; no terraform
  binary on this machine — the files are marked `sketch, not
  deployment-ready` in the header).
- No code changes → full suite still **274 passed, 1 skipped**.
- Cross-checked every mapped component against `configs/default.yaml` and
  the actual module list — nothing built in Steps 1–16 is unmapped.

## Not done (documented in the doc)

VPC/subnet details, DICOM ingestion, auth provider wiring, multi-account
PHI isolation, and the real `S3Storage` adapter — all listed as deferred
decisions a production pass would take.
