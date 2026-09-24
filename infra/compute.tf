# Compute — the Docker image from Step 16 is the unit of deployment.
#
# Local                          Cloud
#   docker build .             -> ECR push (CI does this)
#   medimageforge serve        -> ECS service "api"  behind the ALB
#   medimageforge ui           -> ECS service "ui"   behind the ALB
#   medimageforge ingest|curate|qc|release|train|evaluate
#                              -> ECS/Batch TASKS (run, log, exit)
#
# One image, three run modes: long-running services (api, ui) and batch
# tasks (everything else). The ENTRYPOINT already works this way — a task
# definition just passes the subcommand, e.g. command = ["train"].

resource "aws_ecr_repository" "app" {
  name                 = var.project
  image_tag_mutability = "IMMUTABLE" # tags can't be re-pointed — same idea as release immutability
  image_scanning_configuration { scan_on_push = true }
  tags = local.tags
}

resource "aws_ecs_cluster" "main" {
  name = var.project
  tags = local.tags
}

# --- API service (Step 14): Fargate, private subnets, ALB in front ------------

resource "aws_ecs_task_definition" "api" {
  family                   = "${var.project}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.api_task.arn # deid-only data access — see iam.tf

  container_definitions = jsonencode([{
    name      = "api"
    image     = "${aws_ecr_repository.app.repository_url}:latest"
    command   = ["serve", "--host", "0.0.0.0", "--port", "8000"]
    portMappings = [{ containerPort = 8000 }]
    # Secrets injected as env vars — the code reads config, never files:
    secrets = [{
      name      = "PSEUDONYM_SALT"
      valueFrom = aws_secretsmanager_secret.pseudonym_salt.arn
    }]
    logConfiguration = {
      logDriver = "awslogs"
      options   = { "awslogs-group" = "/ecs/${var.project}", "awslogs-stream-prefix" = "api" }
    }
  }])
  tags = local.tags
}

# --- UI service (Step 15): same pattern ---------------------------------------

resource "aws_ecs_task_definition" "ui" {
  family                   = "${var.project}-ui"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ui_task.arn # NO data-bucket access at all

  container_definitions = jsonencode([{
    name         = "ui"
    image        = "${aws_ecr_repository.app.repository_url}:latest"
    command      = ["ui", "--server.port", "8501"]
    portMappings = [{ containerPort = 8501 }]
    environment  = [{ name = "API_URL", value = "http://api.internal:8000" }]
  }])
  tags = local.tags
}

# --- GPU training: AWS Batch, not ECS -----------------------------------------
# train (Step 10) is CPU-feasible locally because images are 128x128 and the
# dataset is tiny. Real imaging wants a GPU + longer jobs -> Batch manages
# the queue, instance provisioning, and retries.

resource "aws_batch_compute_environment" "gpu" {
  compute_environment_name = "${var.project}-gpu"
  type                     = "MANAGED"
  compute_resources {
    type                = "EC2"
    instance_types      = ["g4dn.xlarge"] # cheapest single-GPU class
    min_vcpus           = 0               # scale to zero — GPUs idle-billed are how budgets die
    max_vcpus           = 16
    subnets             = []              # sketch: private subnet IDs
    instance_role       = ""              # sketch: ecsInstanceRole ARN
  }
  tags = local.tags
}
