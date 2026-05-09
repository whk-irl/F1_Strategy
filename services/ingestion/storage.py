"""S3-compatible object storage client for reading and writing Parquet files.

Supports two backends, selected by ``settings.storage_backend``:

- **minio** (local dev): connects to the docker-compose MinIO instance with
  explicit credentials and a custom endpoint URL.
- **s3** (cloud prod): connects to AWS S3 using the IAM role attached to the
  EKS pod via IRSA — no explicit credentials in the application layer.
  The bucket must already exist (created by Terraform); the app never
  auto-creates S3 buckets to avoid accidental provisioning.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

import boto3
import pandas as pd
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

from .config import IngestionSettings

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

logger = logging.getLogger(__name__)


def _build_s3_client(settings: IngestionSettings) -> "S3Client":
    """Return a boto3 S3 client wired to the correct backend."""
    if settings.storage_backend == "s3":
        # Credentials come from the IAM role / IRSA — never hardcoded.
        return boto3.client(  # type: ignore[return-value]
            "s3",
            region_name=settings.aws_region,
        )
    # MinIO: explicit endpoint + credentials.
    return boto3.client(  # type: ignore[return-value]
        "s3",
        endpoint_url=settings.minio_endpoint_url,
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
        config=BotoConfig(signature_version="s3v4"),
    )


class ObjectStorage:
    """Wraps boto3 S3 operations for Parquet data.

    Args:
        settings: Ingestion configuration.  For the MinIO backend the bucket is
            created on first use; for the S3 backend it must already exist.
    """

    def __init__(self, settings: IngestionSettings) -> None:
        self._bucket = settings.data_bucket
        self._backend = settings.storage_backend
        self._client: S3Client = _build_s3_client(settings)
        self._ensure_bucket()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def write_parquet(self, df: pd.DataFrame, key: str) -> None:
        """Serialise *df* to Parquet and write it to ``s3://<bucket>/<key>``.

        Args:
            df: DataFrame to serialise.
            key: Object key within the bucket (e.g. ``bronze/year=2024/…``).
        """
        buf = io.BytesIO()
        df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
        buf.seek(0)
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=buf.getvalue(),
            ContentType="application/octet-stream",
        )
        logger.info("Wrote %d rows → s3://%s/%s", len(df), self._bucket, key)

    def read_parquet(self, key: str) -> pd.DataFrame:
        """Read a Parquet object from ``s3://<bucket>/<key>`` and return a DataFrame.

        Args:
            key: Object key within the bucket.

        Returns:
            Deserialised DataFrame.

        Raises:
            KeyError: If the object does not exist.
        """
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "NoSuchKey":
                raise KeyError(f"s3://{self._bucket}/{key} does not exist") from exc
            raise
        return pd.read_parquet(io.BytesIO(response["Body"].read()))

    def key_exists(self, key: str) -> bool:
        """Return True if the object at *key* exists in the bucket."""
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "404":
                return False
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_bucket(self) -> None:
        """Verify the bucket exists; auto-create only for the MinIO backend.

        For the S3 backend the bucket must be provisioned by Terraform before
        the pipeline runs.  Auto-creating S3 buckets from application code risks
        naming collisions and bypasses the IaC audit trail.
        """
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code not in ("404", "NoSuchBucket"):
                raise
            if self._backend == "s3":
                raise RuntimeError(
                    f"S3 bucket '{self._bucket}' does not exist. "
                    "Provision it via Terraform (infra/terraform/aws/) before "
                    "running the ingestion pipeline."
                ) from exc
            # MinIO — auto-create for local dev convenience.
            self._client.create_bucket(Bucket=self._bucket)
            logger.info("Created MinIO bucket '%s'", self._bucket)


def build_parquet_key(prefix: str, year: int, round_number: int, session: str) -> str:
    """Return a Hive-partitioned object key.

    Example: ``bronze/year=2024/round=01/session=R/laps.parquet``
    """
    return f"{prefix}/year={year}/round={round_number:02d}/session={session}/laps.parquet"
