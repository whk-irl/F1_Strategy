"""S3-compatible object storage client for reading and writing Parquet files.

Uses boto3 with a configurable endpoint so the same code works against MinIO
in development and any S3-compatible service in production.  The bucket is
created automatically if it does not exist.
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
        settings: Ingestion configuration.  The bucket is created on first use
            if it does not already exist.
    """

    def __init__(self, settings: IngestionSettings) -> None:
        self._bucket = settings.minio_bucket
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
        logger.info("Wrote %d rows to s3://%s/%s", len(df), self._bucket, key)

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
        """Create the bucket if it does not exist (idempotent)."""
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in ("404", "NoSuchBucket"):
                self._client.create_bucket(Bucket=self._bucket)
                logger.info("Created bucket %s", self._bucket)
            else:
                raise


def build_parquet_key(prefix: str, year: int, round_number: int, session: str) -> str:
    """Return a Hive-partitioned object key.

    Example: ``bronze/year=2024/round=1/session=R/laps.parquet``
    """
    return f"{prefix}/year={year}/round={round_number:02d}/session={session}/laps.parquet"
