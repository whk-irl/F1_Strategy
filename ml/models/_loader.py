"""Shared S3/local gold-layer data loader for all ML models."""

from __future__ import annotations

import logging
import os
import pathlib

import pandas as pd
import s3fs

logger = logging.getLogger(__name__)


def load_gold_seasons(seasons: list[int]) -> pd.DataFrame:
    """Read all gold Parquet files for the given seasons from S3, MinIO, or a local file.

    When ``PITWALL_STORAGE_BACKEND=local`` the loader reads a single Parquet file
    at ``PITWALL_LOCAL_DATA_PATH`` (default: ``data/gold_2024.parquet``), ignoring
    the ``seasons`` argument.  This mode is used in the baked Docker image and for
    Streamlit Cloud deployment where S3 is not available.

    Args:
        seasons: Championship years to include (e.g. [2021, 2022, 2023, 2024]).

    Returns:
        Concatenated DataFrame of all gold laps across the requested seasons.

    Raises:
        RuntimeError: If no Parquet files are found for any of the seasons.
    """
    backend = os.getenv("PITWALL_STORAGE_BACKEND", "s3")

    if backend == "local":
        # Anchor candidate paths relative to this file's repo root so that the
        # correct file is found regardless of the process CWD (Streamlit Cloud
        # may run with a different CWD than the repo root).
        _repo_root = pathlib.Path(__file__).parent.parent.parent
        _multi = _repo_root / "data" / "gold_2022_2025.parquet"
        _legacy = _repo_root / "data" / "gold_2024.parquet"
        default_path = str(_multi if _multi.exists() else _legacy)
        local_path = os.getenv("PITWALL_LOCAL_DATA_PATH", default_path)
        if not os.path.exists(local_path):
            raise RuntimeError(
                f"Local data file not found: {local_path}. Run scripts/export_for_deploy.py first."
            )
        df = pd.read_parquet(local_path)
        logger.info("Loaded %d laps from local file %s", len(df), local_path)
        return df
    bucket = os.getenv("PITWALL_AWS_S3_BUCKET", "pitwall-ai-prod")

    if backend == "s3":
        fs = s3fs.S3FileSystem()
    else:
        endpoint = os.getenv("PITWALL_MINIO_ENDPOINT", "localhost:9000")
        bucket = os.getenv("PITWALL_MINIO_BUCKET", "pitwall-ai")
        fs = s3fs.S3FileSystem(
            key=os.getenv("PITWALL_MINIO_ACCESS_KEY", "minioadmin"),
            secret=os.getenv("PITWALL_MINIO_SECRET_KEY", "minioadmin"),
            endpoint_url=f"http://{endpoint}",
        )

    paths: list[str] = []
    for season in seasons:
        prefix = f"{bucket}/gold/year={season}/"
        try:
            found = fs.glob(f"{prefix}**/*.parquet")
            paths.extend(found)
            logger.info("Found %d files for season %d", len(found), season)
        except Exception as exc:
            logger.warning("Could not list %s: %s", prefix, exc)

    if not paths:
        raise RuntimeError(
            f"No gold Parquet files found for seasons {seasons} in s3://{bucket}/gold/. "
            "Run `make ingest-season` first."
        )

    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            with fs.open(path, "rb") as fh:
                frames.append(pd.read_parquet(fh))
        except Exception as exc:
            logger.warning("Skipping %s: %s", path, exc)

    if not frames:
        raise RuntimeError("All Parquet files failed to load.")

    df = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d laps across %d seasons", len(df), len(seasons))
    return df
