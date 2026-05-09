"""Unit tests for services/ingestion/storage.py.

boto3 is mocked throughout — no real S3 / MinIO calls are made.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from botocore.exceptions import ClientError

from services.ingestion.storage import ObjectStorage, build_parquet_key


class TestBuildParquetKey:
    @pytest.mark.parametrize(
        ("prefix", "year", "round_number", "session", "expected"),
        [
            ("bronze", 2024, 1, "R", "bronze/year=2024/round=01/session=R/laps.parquet"),
            ("silver", 2023, 12, "Q", "silver/year=2023/round=12/session=Q/laps.parquet"),
            ("gold", 2024, 5, "FP1", "gold/year=2024/round=05/session=FP1/laps.parquet"),
        ],
    )
    def test_key_format(
        self,
        prefix: str,
        year: int,
        round_number: int,
        session: str,
        expected: str,
    ) -> None:
        assert build_parquet_key(prefix, year, round_number, session) == expected


class TestObjectStorage:
    def test_minio_bucket_created_when_missing(
        self,
        mock_s3: MagicMock,
        settings: "IngestionSettings",  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        """MinIO backend auto-creates a missing bucket (dev convenience)."""
        from services.ingestion.storage import ObjectStorage

        ObjectStorage(settings)
        mock_s3.create_bucket.assert_called_once_with(Bucket="pitwall-test")

    def test_s3_backend_raises_when_bucket_missing(
        self,
        mock_s3: MagicMock,
        settings: "IngestionSettings",  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        """S3 backend must never auto-create buckets — Terraform owns that."""
        from services.ingestion.storage import ObjectStorage

        s3_settings = settings.model_copy(
            update={"storage_backend": "s3", "aws_s3_bucket": "pitwall-prod"}
        )
        with pytest.raises(RuntimeError, match="Terraform"):
            ObjectStorage(s3_settings)

    def test_write_parquet_calls_put_object(
        self,
        mock_s3: MagicMock,
        settings: "IngestionSettings",  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        storage = ObjectStorage(settings)
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        storage.write_parquet(df, "bronze/test.parquet")
        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args.kwargs
        assert call_kwargs["Bucket"] == "pitwall-test"
        assert call_kwargs["Key"] == "bronze/test.parquet"

    def test_read_parquet_deserialises_correctly(
        self,
        mock_s3: MagicMock,
        settings: "IngestionSettings",  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        expected = pd.DataFrame({"x": [10, 20]})
        buf = io.BytesIO()
        expected.to_parquet(buf, index=False)
        buf.seek(0)
        mock_s3.get_object.return_value = {"Body": buf}

        storage = ObjectStorage(settings)
        result = storage.read_parquet("some/key.parquet")
        pd.testing.assert_frame_equal(result, expected)

    def test_read_parquet_missing_key_raises_key_error(
        self,
        mock_s3: MagicMock,
        settings: "IngestionSettings",  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        mock_s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "GetObject"
        )
        storage = ObjectStorage(settings)
        with pytest.raises(KeyError, match="does not exist"):
            storage.read_parquet("missing/key.parquet")

    def test_key_exists_returns_true_when_object_found(
        self,
        mock_s3: MagicMock,
        settings: "IngestionSettings",  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        mock_s3.head_object.return_value = {}  # success
        storage = ObjectStorage(settings)
        assert storage.key_exists("some/key.parquet") is True

    def test_key_exists_returns_false_when_missing(
        self,
        mock_s3: MagicMock,
        settings: "IngestionSettings",  # type: ignore[name-defined]  # noqa: F821
    ) -> None:
        mock_s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
        )
        storage = ObjectStorage(settings)
        assert storage.key_exists("missing/key.parquet") is False
