"""Prefect flow: FastF1 → Bronze → Silver → Gold.

Entry point for scheduled and on-demand ingestion runs.

Usage (CLI):
    python -m services.ingestion.pipeline --year 2024 --round 1 --session R

Usage (Prefect deployment):
    Triggered via ``make ingest`` or the Prefect UI.
"""

from __future__ import annotations

import logging
from typing import Annotated

import mlflow
import typer
from prefect import flow, task
from prefect.artifacts import create_table_artifact
from prefect.cache_policies import NO_CACHE

from .config import IngestionSettings
from .fastf1_client import FastF1Client
from .schemas.bronze import BronzeLapSchema
from .storage import ObjectStorage, build_parquet_key
from .transforms import bronze_to_silver, silver_to_gold

logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

# ---------------------------------------------------------------------------
# Prefect tasks
# ---------------------------------------------------------------------------


@task(name="fetch-bronze", retries=2, retry_delay_seconds=30, cache_policy=NO_CACHE)
def fetch_and_store_bronze(
    client: FastF1Client,
    storage: ObjectStorage,
    settings: IngestionSettings,
    year: int,
    round_number: int,
    session: str,
) -> str:
    """Fetch raw laps from FastF1, validate the bronze schema, and persist to MinIO.

    Args:
        client: Configured FastF1 client.
        storage: Configured object storage client.
        settings: Ingestion settings (used for the storage prefix).
        year: Championship year.
        round_number: Race round number.
        session: Session type (e.g. ``'R'``).

    Returns:
        The S3 key of the written Parquet object.
    """
    raw_df = client.get_session_laps(year, round_number, session)

    # Validate against the bronze schema before persisting.
    BronzeLapSchema.validate(raw_df)

    key = build_parquet_key(settings.bronze_prefix, year, round_number, session)
    storage.write_parquet(raw_df, key)
    logger.info("Bronze written → s3://%s/%s", settings.data_bucket, key)
    return key


@task(name="transform-to-silver", cache_policy=NO_CACHE)
def transform_to_silver_task(
    storage: ObjectStorage,
    settings: IngestionSettings,
    bronze_key: str,
    year: int,
    round_number: int,
    session: str,
) -> str:
    """Read bronze, transform to silver, validate, and persist.

    Args:
        storage: Object storage client.
        settings: Ingestion settings.
        bronze_key: S3 key of the bronze Parquet object.
        year: Championship year.
        round_number: Race round number.
        session: Session type.

    Returns:
        The S3 key of the silver Parquet object.
    """
    bronze_df = storage.read_parquet(bronze_key)
    silver_df = bronze_to_silver(bronze_df, year=year, round_number=round_number, session=session)

    key = build_parquet_key(settings.silver_prefix, year, round_number, session)
    storage.write_parquet(silver_df, key)
    logger.info("Silver written → s3://%s/%s  (%d rows)", settings.data_bucket, key, len(silver_df))
    return key


@task(name="transform-to-gold", cache_policy=NO_CACHE)
def transform_to_gold_task(
    storage: ObjectStorage,
    settings: IngestionSettings,
    silver_key: str,
    year: int,
    round_number: int,
    session: str,
    total_laps: int,
) -> str:
    """Read silver, engineer gold features, validate, and persist.

    Args:
        storage: Object storage client.
        settings: Ingestion settings.
        silver_key: S3 key of the silver Parquet object.
        year: Championship year.
        round_number: Race round number.
        session: Session type.
        total_laps: Scheduled race length (for ``race_progress`` normalisation).

    Returns:
        The S3 key of the gold Parquet object.
    """
    silver_df = storage.read_parquet(silver_key)
    gold_df = silver_to_gold(silver_df, total_laps=total_laps)

    key = build_parquet_key(settings.gold_prefix, year, round_number, session)
    storage.write_parquet(gold_df, key)
    logger.info("Gold written → s3://%s/%s  (%d rows)", settings.data_bucket, key, len(gold_df))
    return key


# ---------------------------------------------------------------------------
# Prefect flow
# ---------------------------------------------------------------------------


@flow(
    name="f1-ingestion",
    description="Ingest one F1 session: FastF1 → bronze → silver → gold Parquet on MinIO.",
    version="1.0.0",
)
def ingest_session(
    year: int,
    round_number: int,
    session: str = "R",
    total_laps: int = 57,
) -> dict[str, str]:
    """End-to-end ingestion flow for a single F1 session.

    Args:
        year: Championship year.
        round_number: Round number within the season.
        session: Session type (``'R'``, ``'Q'``, ``'FP1'``, etc.).
        total_laps: Scheduled race length in laps (used only for gold features).

    Returns:
        A dict mapping layer names to the S3 keys written.
    """
    settings = IngestionSettings()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=f"ingest-{year}-R{round_number:02d}-{session}"):
        mlflow.log_params({"year": year, "round_number": round_number, "session": session})

        client = FastF1Client(settings)
        storage = ObjectStorage(settings)

        bronze_key = fetch_and_store_bronze(client, storage, settings, year, round_number, session)
        silver_key = transform_to_silver_task(
            storage, settings, bronze_key, year, round_number, session
        )
        gold_key = transform_to_gold_task(
            storage, settings, silver_key, year, round_number, session, total_laps
        )

        keys = {"bronze": bronze_key, "silver": silver_key, "gold": gold_key}
        mlflow.log_params({f"{layer}_key": key for layer, key in keys.items()})

    # Surface the artefact keys in the Prefect UI.
    create_table_artifact(
        key="layer-keys",
        table=[{"layer": k, "s3_key": v} for k, v in keys.items()],
        description=f"Storage keys for {year} R{round_number} {session}",
    )

    return keys


# ---------------------------------------------------------------------------
# CLI entry point (python -m services.ingestion.pipeline)
# ---------------------------------------------------------------------------


@app.command()
def main(
    year: Annotated[int, typer.Option("--year", help="Championship year.")] = 2024,
    round: Annotated[int, typer.Option("--round", help="Race round number.")] = 1,
    session: Annotated[str, typer.Option("--session", help="Session type (R/Q/FP1…).")] = "R",
    total_laps: Annotated[int, typer.Option("--total-laps", help="Scheduled race laps.")] = 57,
    all_rounds: Annotated[
        bool, typer.Option("--all-rounds", help="Ingest every round in the season.")
    ] = False,
) -> None:
    """CLI wrapper around the ``ingest_session`` Prefect flow."""
    if all_rounds:
        import fastf1

        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for _, event in schedule.iterrows():
            ingest_session(
                year=year,
                round_number=int(event["RoundNumber"]),
                session=session,
                total_laps=total_laps,
            )
    else:
        ingest_session(year=year, round_number=round, session=session, total_laps=total_laps)


if __name__ == "__main__":
    app()
