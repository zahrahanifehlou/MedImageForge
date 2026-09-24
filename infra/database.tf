# Database — local artifacts/manifest.db (SQLite) becomes managed Postgres.
#
# Why not keep SQLite? SQLite is a file: one writer at a time, no network
# access control, no concurrent API readers while a pipeline writes. The
# manifest is the platform's source of truth AND the API's backend — that's
# a server database's job. Code impact: manifest.py's SQL is deliberately
# simple (SQLAlchemy-free but standard SQL); the migration is a dialect
# port, not a redesign. Slices/labels/annotations/curation/patients tables
# all carry over.

resource "aws_db_instance" "manifest" {
  identifier            = "${var.project}-manifest"
  engine                = "postgres"
  engine_version        = "16"
  instance_class        = "db.t4g.micro" # learning-scale; a real registry grows this
  allocated_storage     = 20
  storage_encrypted     = true
  kms_key_id            = aws_kms_key.platform.arn
  db_name               = "manifest"
  username              = "manifest_admin"
  manage_master_user_password = true    # credentials land in Secrets Manager
  skip_final_snapshot   = false
  deletion_protection   = true
  backup_retention_period = 7
  publicly_accessible   = false         # private subnet only
  tags                  = local.tags
}
