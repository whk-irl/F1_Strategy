"""Runtime configuration for the ingestion service.

All settings are read from environment variables (prefixed ``PITWALL_``) or
from a ``.env`` file at the project root.

**Backends:**
- ``storage_backend=minio`` (default) — local docker-compose stack.
- ``storage_backend=s3`` — AWS S3; credentials come from the IAM role attached
  to the EKS node / pod (IRSA).  No explicit keys needed.

See ``.env.example`` for a full variable reference.
"""

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestionSettings(BaseSettings):
    """Pydantic-settings model for the ingestion service."""

    model_config = SettingsConfigDict(
        env_prefix="PITWALL_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # FastF1 cache — keeps session data on disk so repeated runs don't re-download.
    fastf1_cache_path: str = Field(
        default=".fastf1_cache",
        description="Directory for FastF1 HTTP cache (excluded from git).",
    )

    # ---------------------------------------------------------------------------
    # Storage backend
    # ---------------------------------------------------------------------------
    storage_backend: Literal["minio", "s3"] = Field(
        default="minio",
        description=(
            "'minio' for local docker-compose dev stack; "
            "'s3' for AWS S3 (credentials via IAM role / IRSA)."
        ),
    )

    # MinIO (local dev only — ignored when storage_backend='s3').
    minio_endpoint: str = Field(
        default="localhost:9000",
        description="Host:port of the local MinIO endpoint.",
    )
    minio_access_key: str = Field(default="minioadmin")
    minio_secret_key: str = Field(default="minioadmin")
    minio_bucket: str = Field(
        default="pitwall-ai",
        description="Bucket name for the local MinIO dev stack.",
    )
    minio_use_ssl: bool = Field(default=False)

    # AWS S3 (prod / cloud — ignored when storage_backend='minio').
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region for S3 and ECR.",
    )
    aws_s3_bucket: str = Field(
        default="pitwall-ai-prod",
        description="S3 bucket name (created by Terraform; never auto-created by the app).",
    )

    # Storage key prefixes — Hive-style partitioning within the bucket.
    bronze_prefix: str = Field(default="bronze")
    silver_prefix: str = Field(default="silver")
    gold_prefix: str = Field(default="gold")

    # MLflow tracking server.
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_experiment_name: str = Field(default="pitwall-ingestion")

    # ---------------------------------------------------------------------------
    # Derived properties
    # ---------------------------------------------------------------------------

    @property
    def data_bucket(self) -> str:
        """Return the active bucket name for the current storage backend."""
        return self.aws_s3_bucket if self.storage_backend == "s3" else self.minio_bucket

    @property
    def minio_endpoint_url(self) -> str:
        """Full URL for boto3 ``endpoint_url`` (MinIO only)."""
        scheme = "https" if self.minio_use_ssl else "http"
        return f"{scheme}://{self.minio_endpoint}"

    # ---------------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------------

    @model_validator(mode="after")
    def _check_s3_bucket_name(self) -> "IngestionSettings":
        if self.storage_backend == "s3" and not self.aws_s3_bucket:
            raise ValueError("PITWALL_AWS_S3_BUCKET must be set when storage_backend='s3'.")
        return self
