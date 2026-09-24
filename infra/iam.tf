# IAM — the privacy boundary, enforced by infrastructure instead of code.
#
# On localhost, "the API only speaks pseudonyms" is a convention in api.py.
# Here it is a fact: the API/UI task roles CANNOT reach the raw bucket or
# the real-patient-ID columns — there is no policy granting it. A bug in
# the code can no longer leak what it cannot read.

resource "aws_iam_role" "ecs_execution" {
  name = "${var.project}-ecs-execution"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
  tags = local.tags
}

# Pipeline role: ingest/curate/qc/release tasks — read raw, write curated.
# The ONLY role allowed to read real patient IDs from the manifest.
resource "aws_iam_role" "pipeline_task" {
  name = "${var.project}-pipeline-task"
  assume_role_policy = aws_iam_role.ecs_execution.assume_role_policy
  tags               = local.tags
}

resource "aws_iam_role_policy" "pipeline_data" {
  role = aws_iam_role.pipeline_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:ListBucket"]
        Resource = [aws_s3_bucket.raw.arn, "${aws_s3_bucket.raw.arn}/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.curated.arn,   "${aws_s3_bucket.curated.arn}/*",
          aws_s3_bucket.deid.arn,      "${aws_s3_bucket.deid.arn}/*",
          aws_s3_bucket.releases.arn,  "${aws_s3_bucket.releases.arn}/*",
          aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*",
        ]
      },
      {
        Effect   = "Allow" # the salt: only the privacy task reads it
        Action   = "secretsmanager:GetSecretValue"
        Resource = aws_secretsmanager_secret.pseudonym_salt.arn
      },
    ]
  })
}

# API role: curated + deid + releases + artifacts, NO raw bucket.
# Even if api.py had a bug that tried to serve a raw path, S3 denies it.
resource "aws_iam_role" "api_task" {
  name = "${var.project}-api-task"
  assume_role_policy = aws_iam_role.ecs_execution.assume_role_policy
  tags               = local.tags
}

resource "aws_iam_role_policy" "api_data" {
  role = aws_iam_role.api_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:ListBucket"]
      Resource = [
        aws_s3_bucket.curated.arn,   "${aws_s3_bucket.curated.arn}/*",
        aws_s3_bucket.deid.arn,      "${aws_s3_bucket.deid.arn}/*",
        aws_s3_bucket.releases.arn,  "${aws_s3_bucket.releases.arn}/*",
        aws_s3_bucket.artifacts.arn, "${aws_s3_bucket.artifacts.arn}/*",
      ]
    }] # deliberately NO raw.arn, NO pseudonym_salt secret
  })
}

# UI role: nothing. The UI talks HTTP to the API only (Step 15's contract —
# "the UI uses only this API" — becomes literally true at the network level).
resource "aws_iam_role" "ui_task" {
  name = "${var.project}-ui-task"
  assume_role_policy = aws_iam_role.ecs_execution.assume_role_policy
  tags               = local.tags
}
