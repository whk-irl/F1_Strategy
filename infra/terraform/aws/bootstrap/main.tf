# Bootstrap — creates the S3 bucket and DynamoDB table that Terraform uses
# to store its own state for the main infra/terraform/aws/ module.
#
# Run ONCE with local state before initialising the main module:
#   terraform -chdir=infra/terraform/aws/bootstrap init
#   terraform -chdir=infra/terraform/aws/bootstrap apply -auto-approve
#
# After apply, copy the outputs into infra/terraform/aws/backend.tf.

terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.50"
    }
  }
  # Intentionally local state — this module itself is tiny and infrequently
  # changed, so remote state is unnecessary overhead.
}

provider "aws" {
  region = var.aws_region
}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Short project identifier used in resource names."
  type        = string
  default     = "pitwall-ai"
}

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

data "aws_caller_identity" "current" {}

# ---------------------------------------------------------------------------
# S3 state bucket
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "tfstate" {
  # Account ID suffix ensures global uniqueness.
  bucket = "${var.project_name}-tfstate-${data.aws_caller_identity.current.account_id}"

  # Prevent accidental deletion of the state bucket.
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_versioning" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "tfstate" {
  bucket                  = aws_s3_bucket.tfstate.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------------------------------------------------------------------
# DynamoDB lock table
# ---------------------------------------------------------------------------

resource "aws_dynamodb_table" "tflock" {
  name         = "${var.project_name}-tflock"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }
}

# ---------------------------------------------------------------------------
# Outputs — paste these into infra/terraform/aws/backend.tf
# ---------------------------------------------------------------------------

output "state_bucket" {
  description = "Name of the S3 bucket storing Terraform state."
  value       = aws_s3_bucket.tfstate.bucket
}

output "lock_table" {
  description = "Name of the DynamoDB table used for state locking."
  value       = aws_dynamodb_table.tflock.name
}

output "backend_tf_snippet" {
  description = "Paste this block into infra/terraform/aws/backend.tf."
  value       = <<-EOT
    terraform {
      backend "s3" {
        bucket         = "${aws_s3_bucket.tfstate.bucket}"
        key            = "pitwall-ai/terraform.tfstate"
        region         = "${var.aws_region}"
        dynamodb_table = "${aws_dynamodb_table.tflock.name}"
        encrypt        = true
      }
    }
  EOT
}
