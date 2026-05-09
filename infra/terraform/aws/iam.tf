# IAM — IRSA role for the pitwall-training Kubernetes service account.
#
# This role is bound to the `pitwall-training` service account in the
# `pitwall` namespace.  It grants read/write access to the S3 data lake
# and read access to ECR, so training pods can pull their own images and
# read/write feature data without any hardcoded credentials.

# ---------------------------------------------------------------------------
# S3 data-lake access policy
# ---------------------------------------------------------------------------

resource "aws_iam_policy" "s3_data_lake" {
  name        = "${local.cluster_name}-s3-data-lake"
  description = "Read/write access to the Pitwall AI S3 data lake."

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ListBucket"
        Effect = "Allow"
        Action = ["s3:ListBucket", "s3:GetBucketLocation"]
        Resource = [aws_s3_bucket.data_lake.arn]
      },
      {
        Sid    = "ReadWriteObjects"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:GetObjectVersion",
        ]
        Resource = ["${aws_s3_bucket.data_lake.arn}/*"]
      }
    ]
  })
}

# ---------------------------------------------------------------------------
# MLflow artefact access (same bucket, mlruns/ prefix)
# Included in the same policy above — no separate policy needed.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# IRSA role — pitwall-training service account
# ---------------------------------------------------------------------------

module "pitwall_training_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.39"

  role_name = "${local.cluster_name}-training"

  oidc_providers = {
    main = {
      provider_arn = module.eks.oidc_provider_arn
      # Binds to the `pitwall-training` service account in the `pitwall` namespace.
      namespace_service_accounts = ["pitwall:pitwall-training"]
    }
  }

  role_policy_arns = {
    s3  = aws_iam_policy.s3_data_lake.arn
    ecr = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  }
}

# ---------------------------------------------------------------------------
# IRSA role — pitwall-ingestion service account
# (Same permissions as training; separate role for auditability.)
# ---------------------------------------------------------------------------

module "pitwall_ingestion_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.39"

  role_name = "${local.cluster_name}-ingestion"

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["pitwall:pitwall-ingestion"]
    }
  }

  role_policy_arns = {
    s3 = aws_iam_policy.s3_data_lake.arn
  }
}

# ---------------------------------------------------------------------------
# Kubernetes service accounts (created in the cluster)
# The IRSA annotation tells the pod mutation webhook to inject AWS credentials.
# ---------------------------------------------------------------------------

resource "kubernetes_namespace" "pitwall" {
  metadata {
    name = "pitwall"
    labels = {
      "app.kubernetes.io/managed-by" = "terraform"
    }
  }

  depends_on = [module.eks]
}

resource "kubernetes_service_account" "training" {
  metadata {
    name      = "pitwall-training"
    namespace = kubernetes_namespace.pitwall.metadata[0].name
    annotations = {
      "eks.amazonaws.com/role-arn" = module.pitwall_training_irsa.iam_role_arn
    }
  }
}

resource "kubernetes_service_account" "ingestion" {
  metadata {
    name      = "pitwall-ingestion"
    namespace = kubernetes_namespace.pitwall.metadata[0].name
    annotations = {
      "eks.amazonaws.com/role-arn" = module.pitwall_ingestion_irsa.iam_role_arn
    }
  }
}
