"""Shared pytest fixtures for the Pitwall AI test suite.

Integration tests that require MinIO or MLflow are marked with
``@pytest.mark.integration`` and skipped in unit-only runs
(``pytest tests/unit``).
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from services.ingestion.config import IngestionSettings


# ---------------------------------------------------------------------------
# Minimal settings fixture pointing at test doubles
# ---------------------------------------------------------------------------


@pytest.fixture()
def settings() -> IngestionSettings:
    """Ingestion settings wired to test defaults (no real services required)."""
    return IngestionSettings(
        fastf1_cache_path=".fastf1_cache_test",
        minio_endpoint="localhost:9000",
        minio_access_key="test",
        minio_secret_key="test",
        minio_bucket="pitwall-test",
        minio_use_ssl=False,
        mlflow_tracking_uri="http://localhost:5000",
    )


# ---------------------------------------------------------------------------
# Synthetic bronze DataFrame (mirrors FastF1's session.laps columns)
# ---------------------------------------------------------------------------


@pytest.fixture()
def bronze_laps_df() -> pd.DataFrame:
    """Minimal synthetic bronze DataFrame covering the common case.

    Includes one pit stop, one SC lap, and one inaccurate lap so that
    transform tests exercise the branching logic.
    """
    n = 10
    return pd.DataFrame(
        {
            "Driver": ["VER"] * n,
            "DriverNumber": ["1"] * n,
            "LapNumber": list(range(1, n + 1)),
            "Stint": [1] * 5 + [2] * 5,
            "LapTime": pd.to_timedelta(
                [90.5, 91.2, 91.8, 92.1, pd.NaT, 89.5, 89.8, 90.0, 90.3, 90.6],
                unit="s",
            ),
            "Sector1Time": pd.to_timedelta([30.0] * n, unit="s"),
            "Sector2Time": pd.to_timedelta([30.5] * n, unit="s"),
            "Sector3Time": pd.to_timedelta([30.0] * n, unit="s"),
            "Compound": ["MEDIUM"] * 5 + ["HARD"] * 5,
            "TyreLife": list(range(1, 6)) + list(range(1, 6)),
            "FreshTyre": [True] + [False] * 9,
            "Position": [3.0] * n,
            "TrackStatus": ["1"] * 3 + ["6"] * 2 + ["1"] * 5,
            "IsAccurate": [True] * 4 + [False] + [True] * 5,
            # Pit timing (only on the pit lap, lap 5)
            "PitInTime": [pd.NaT] * 4
            + [pd.Timedelta("45 min")]
            + [pd.NaT] * 5,
            "PitOutTime": [pd.NaT] * 5
            + [pd.Timedelta("46 min")]
            + [pd.NaT] * 4,
        }
    )


# ---------------------------------------------------------------------------
# Synthetic silver DataFrame
# ---------------------------------------------------------------------------


@pytest.fixture()
def silver_laps_df(bronze_laps_df: pd.DataFrame) -> pd.DataFrame:
    """Return silver laps derived from the bronze fixture via the real transform."""
    from services.ingestion.transforms import bronze_to_silver

    return bronze_to_silver(bronze_laps_df, year=2024, round_number=1, session="R")


# ---------------------------------------------------------------------------
# Mock S3 / boto3 for unit tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_s3(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch boto3.client so ObjectStorage never touches real AWS/MinIO."""
    mock_client = MagicMock()

    # head_bucket raises 404 on first call (bucket not found), succeeds after create.
    from botocore.exceptions import ClientError

    mock_client.head_bucket.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadBucket"
    )
    mock_client.create_bucket.return_value = {}

    with patch("services.ingestion.storage.boto3.client", return_value=mock_client):
        yield mock_client
