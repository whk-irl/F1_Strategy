# ECR repositories for training and serving container images.

locals {
  ecr_repos = {
    "pitwall-tire-degradation" = "CPU training image for the tire degradation LightGBM model."
    "pitwall-safety-car"       = "CPU training image for the safety-car probability model."
    "pitwall-strategy-policy"  = "GPU training image for the PPO strategy policy (CUDA 12.1)."
    "pitwall-ingestion"        = "Ingestion pipeline image (FastF1 → S3)."
    "pitwall-api"              = "BentoML / FastAPI inference service."
  }
}

resource "aws_ecr_repository" "training" {
  for_each = local.ecr_repos

  name                 = each.key
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Lifecycle policy: keep the last 10 tagged images + delete untagged images
# older than 7 days to limit storage costs.
resource "aws_ecr_lifecycle_policy" "training" {
  for_each   = aws_ecr_repository.training
  repository = each.value.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Remove untagged images older than 7 days"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 7
        }
        action = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "Keep last 10 tagged images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v", "latest", "sha-"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = { type = "expire" }
      }
    ]
  })
}
