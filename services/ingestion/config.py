"""Runtime configuration for the ingestion service.

All settings are read from environment variables (prefixed ``PITWALL_``) or
from a ``.env`` file at the project root.  Sensible defaults are provided for
local development (docker-compose stack).
"""

from pydantic import Field
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

    # Object storage (MinIO in dev, any S3-compatible service in prod).
    minio_endpoint: str = Field(
        default="localhost:9000",
        description="Host:port of the S3-compatible endpoint.",
    )
    minio_access_key: str = Field(default="minioadmin")
    minio_secret_key: str = Field(default="minioadmin")
    minio_bucket: str = Field(
        default="pitwall-ai",
        description="Bucket that holds all bronze/silver/gold data.",
    )
    minio_use_ssl: bool = Field(default=False)

    # Storage key prefixes — follow Hive-style partitioning within the bucket.
    bronze_prefix: str = Field(default="bronze")
    silver_prefix: str = Field(default="silver")
    gold_prefix: str = Field(default="gold")

    # MLflow tracking server.
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_experiment_name: str = Field(default="pitwall-ingestion")

    @property
    def minio_endpoint_url(self) -> str:
        """Full URL for boto3 ``endpoint_url`` parameter."""
        scheme = "https" if self.minio_use_ssl else "http"
        return f"{scheme}://{self.minio_endpoint}"
