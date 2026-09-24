output "buckets" {
  value = {
    raw       = aws_s3_bucket.raw.bucket
    curated   = aws_s3_bucket.curated.bucket
    deid      = aws_s3_bucket.deid.bucket
    releases  = aws_s3_bucket.releases.bucket
    artifacts = aws_s3_bucket.artifacts.bucket
  }
}

output "ecr_repository" { value = aws_ecr_repository.app.repository_url }
output "manifest_db_endpoint" { value = aws_db_instance.manifest.endpoint }
output "pipeline_state_machine" { value = aws_sfn_state_machine.pipeline.arn }
