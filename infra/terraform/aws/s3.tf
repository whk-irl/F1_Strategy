# S3 data lake — bronze / silver / gold Parquet storage.

resource "aws_s3_bucket" "data_lake" {
  bucket = "${var.project_name}-${var.environment}"  # e.g. pitwall-ai-prod
}

resource "aws_s3_bucket_versioning" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "data_lake" {
  bucket                  = aws_s3_bucket.data_lake.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id

  # Bronze: raw FastF1 dumps.  Move to Glacier after 90 days to cut storage
  # cost — bronze is only needed if we need to re-run the silver transform.
  rule {
    id     = "bronze-to-glacier"
    status = "Enabled"
    filter { prefix = "bronze/" }
    transition {
      days          = var.bronze_retention_days
      storage_class = "GLACIER"
    }
  }

  # Silver and gold are small; keep them in S3 Standard indefinitely.
  rule {
    id     = "silver-noop"
    status = "Enabled"
    filter { prefix = "silver/" }
    noncurrent_version_expiration { noncurrent_days = 30 }
  }

  rule {
    id     = "gold-noop"
    status = "Enabled"
    filter { prefix = "gold/" }
    noncurrent_version_expiration { noncurrent_days = 30 }
  }

  # MLflow artefacts — expire old versions after 180 days.
  rule {
    id     = "mlruns-noncurrent-expiry"
    status = "Enabled"
    filter { prefix = "mlruns/" }
    noncurrent_version_expiration { noncurrent_days = 180 }
  }
}
