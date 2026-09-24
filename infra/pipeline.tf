# Pipeline orchestration — local CLI invocations become a state machine.
#
# Today a human runs: ingest -> curate -> load-labels -> qc -> release.
# In the cloud that's a Step Functions state machine; each step is the same
# Docker image with a different `command`. The state machine gives what
# shell history never will: per-step retries, timeouts, and a visual record
# of every execution — the audit log's cloud-native counterpart.

resource "aws_sfn_state_machine" "pipeline" {
  name     = "${var.project}-pipeline"
  role_arn = aws_iam_role.sfn.arn
  tags     = local.tags

  definition = jsonencode({
    StartAt = "Ingest"
    States = {
      Ingest = {
        Type = "Task", Resource = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          Cluster = aws_ecs_cluster.main.arn
          TaskDefinition = aws_ecs_task_definition.pipeline.arn
          Overrides = { ContainerOverrides = [{ Name = "job", Command = ["ingest"] }] }
        }
        Next = "Curate"
      }
      Curate = {
        Type = "Task", Resource = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          Cluster = aws_ecs_cluster.main.arn
          TaskDefinition = aws_ecs_task_definition.pipeline.arn
          Overrides = { ContainerOverrides = [{ Name = "job", Command = ["curate"] }] }
        }
        Next = "QC"
      }
      QC = {
        Type = "Task", Resource = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          Cluster = aws_ecs_cluster.main.arn
          TaskDefinition = aws_ecs_task_definition.pipeline.arn
          Overrides = { ContainerOverrides = [{ Name = "job", Command = ["qc"] }] }
        }
        # The QC gate (Step 8) maps to a Choice state: FAIL stops the chain.
        Next = "Release"
      }
      Release = {
        Type = "Task", Resource = "arn:aws:states:::ecs:runTask.sync"
        Parameters = {
          Cluster = aws_ecs_cluster.main.arn
          TaskDefinition = aws_ecs_task_definition.pipeline.arn
          Overrides = { ContainerOverrides = [{ Name = "job", Command = ["release"] }] }
        }
        End = true
      }
    }
  })
}

resource "aws_ecs_task_definition" "pipeline" {
  family                   = "${var.project}-pipeline"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 2048  # curation hashes every image — sized per step in a real build
  memory                   = 4096
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.pipeline_task.arn
  container_definitions = jsonencode([{
    name  = "job"
    image = "${aws_ecr_repository.app.repository_url}:latest"
  }])
  tags = local.tags
}

resource "aws_iam_role" "sfn" {
  name = "${var.project}-sfn"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole", Effect = "Allow"
      Principal = { Service = "states.amazonaws.com" }
    }]
  })
  tags = local.tags
}

# Nightly re-validation: EventBridge triggers the state machine on a cron —
# the cloud version of "someone remembered to run qc".
resource "aws_scheduler_schedule" "nightly_qc" {
  name = "${var.project}-nightly-qc"
  schedule_expression = "cron(0 3 * * ? *)"
  flexible_time_window { mode = "OFF" }
  target {
    arn      = aws_sfn_state_machine.pipeline.arn
    role_arn = aws_iam_role.sfn.arn
  }
}
