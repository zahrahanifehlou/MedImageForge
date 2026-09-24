# Storage zones — one bucket per local directory.
#
# Local (configs/default.yaml)        Cloud
#   data/Patients_CT/               -> s3://...-raw      (never modified)
#   artifacts/curated/              -> s3://...-curated
#   artifacts/deid/                 -> s3://...-deid     (only zone that may leave)
#   datasets/                       -> s3://...-releases (immutable snapshots)
#   artifacts/ (runs, eval, audit)  -> s3://...-artifacts

resource "aws_s3_bucket" "raw" {
  bucket = "${var.project}-raw"
  tags   = local.tags
}

resource "aws_s3_bucket" "curated" {
  bucket = "${var.project}-curated"
  tags   = local.tags
}

resource "aws_s3_bucket" "deid" {
  bucket = "${var.project}-deid"
  tags   = local.tags
}

resource "aws_s3_bucket" "releases" {
  bucket = "${var.project}-releases"
  tags   = local.tags
}

resource "aws_s3_bucket" "artifacts" {
  bucket = "${var.project}-artifacts"
  tags   = local.tags
}

# --- every bucket: encrypted, private, versioned ------------------------------

resource "aws_s3_bucket_server_side_encryption_configuration" "all" {
  for_each = {
    raw = aws_s3_bucket.raw.id, curated = aws_s3_bucket.curated.id,
    deid = aws_s3_bucket.deid.id, releases = aws_s3_bucket.releases.id,
    artifacts = aws_s3_bucket.artifacts.id,
  }
  bucket = each.value
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "aws:kms" }
  }
}

resource "aws_s3_bucket_public_access_block" "all" {
  for_each                = aws_s3_bucket_server_side_encryption_configuration.all
  bucket                  = each.value.bucket
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "all" {
  for_each = aws_s3_bucket_server_side_encryption_configuration.all
  bucket   = each.value.bucket
  versioning_configuration { status = "Enabled" }
}

# --- the two special policies --------------------------------------------------

# releases/: the local rule "a published dataset is immutable" becomes
# Object Lock — objects cannot be deleted or overwritten, even by admins,
# for the retention period. Same guarantee Step 9 enforced in Python.
resource "aws_s3_bucket_object_lock_configuration" "releases" {
  bucket = aws_s3_bucket.releases.id
  rule {
    default_retention {
      mode = "GOVERNANCE" # COMPLIANCE for regulated deployments
      days = 2555         # ~7 years, typical imaging retention
    }
  }
}

# artifacts/audit.log: the local hash chain is already tamper-evident;
# Object Lock makes it tamper-PROOF — the platform cannot edit its own log.
resource "aws_s3_bucket_object_lock_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    default_retention {
      mode = "GOVERNANCE"
      days = 2555
    }
  }
}
