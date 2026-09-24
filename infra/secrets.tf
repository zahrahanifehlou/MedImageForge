# Secrets — local .secrets/pseudonym_salt becomes a managed secret.
#
# The salt is the re-identification key (Step 6): whoever holds it can map
# PAT-xxxx back to patient 049. Locally it was a gitignored file; here it
# lives in Secrets Manager with rotation, and ONLY the privacy/deid task
# role can read it — enforced by IAM, not by a .gitignore file.

resource "aws_secretsmanager_secret" "pseudonym_salt" {
  name       = "${var.project}/pseudonym-salt"
  kms_key_id = aws_kms_key.platform.arn
  tags       = local.tags
}

resource "aws_kms_key" "platform" {
  description         = "${var.project} encryption key (buckets, secrets, DB)"
  enable_key_rotation = true
  tags                = local.tags
}
