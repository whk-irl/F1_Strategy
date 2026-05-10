"""Tire degradation model — LightGBM regression training.

Target:  ``lap_time_delta_s`` (driver lap time minus their stint median).
         Modelling the *delta* removes car/driver baseline pace and isolates
         tyre degradation as a function of compound and tyre age.

Entry point:
    python -m ml.models.tire_degradation.train
    make train-tire
"""

from __future__ import annotations

import logging
from typing import Annotated

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import pandas as pd
import typer
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sklearn.metrics import mean_absolute_error, r2_score

from ml.models._loader import load_gold_seasons

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

app = typer.Typer(add_completion=False)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [
    "tyre_life_laps",
    "compound_encoded",
    "race_progress",
    "track_status_encoded",
    "is_fresh_tyre",
    "lap_delta_to_field_median_s",  # captures car pace vs field — acts as car baseline
]

TARGET_COL = "lap_time_delta_s"


class TireDegConfig(BaseSettings):
    """Hyper-parameters and data settings for tire degradation training."""

    model_config = SettingsConfigDict(env_prefix="PITWALL_TIRE_", env_file=".env", extra="ignore")

    training_seasons: list[int] = Field(default=[2024])
    val_rounds: list[int] = Field(default=[5, 10, 15, 20])

    n_estimators: int = Field(default=1000)
    learning_rate: float = Field(default=0.05)
    num_leaves: int = Field(default=63)
    min_child_samples: int = Field(default=30)
    subsample: float = Field(default=0.8)
    colsample_bytree: float = Field(default=0.8)
    early_stopping_rounds: int = Field(default=50)

    mlflow_tracking_uri: str = Field(
        default="mlruns", validation_alias="PITWALL_MLFLOW_TRACKING_URI"
    )
    mlflow_experiment: str = Field(default="tire-degradation")
    registered_model_name: str = Field(default="pitwall-tire-degradation")


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------


def prepare_features(
    df: pd.DataFrame,
    val_rounds: list[int],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Filter laps and split into train / validation sets.

    Args:
        df: Raw gold DataFrame (all seasons).
        val_rounds: Round numbers held out for validation.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val).
    """
    # Keep only accurate race laps on a clear track — no pit laps, no SC noise.
    mask = (
        (df["session"] == "R")
        & (df["is_accurate"].astype(bool))
        & (df["track_status_encoded"] == 0)
        & (~df["pit_in_this_lap"].astype(bool))
        & (~df["pit_out_this_lap"].astype(bool))
        & df[TARGET_COL].notna()
        & df["tyre_life_laps"].notna()
        & (df["tyre_life_laps"] >= 2)  # exclude out-laps with warm-up noise
        & df["lap_delta_to_field_median_s"].notna()
    )
    clean = df[mask].copy()
    clean["is_fresh_tyre"] = clean["is_fresh_tyre"].fillna(False).astype(int)

    val_mask = clean["round_number"].isin(val_rounds)
    train = clean[~val_mask]
    val = clean[val_mask]

    logger.info(
        "Dataset: %d train laps | %d val laps (rounds %s held out)",
        len(train),
        len(val),
        val_rounds,
    )

    if len(train) < 100:
        logger.warning(
            "Training set is very small (%d rows). Ingest more seasons for a robust model.",
            len(train),
        )

    return (
        train[FEATURE_COLS],
        train[TARGET_COL],
        val[FEATURE_COLS],
        val[TARGET_COL],
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(config: TireDegConfig | None = None) -> None:
    """Train the tire degradation model and register it in MLflow.

    Args:
        config: Training configuration (defaults to env-var / defaults).
    """
    if config is None:
        config = TireDegConfig()

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment)

    df = load_gold_seasons(config.training_seasons)
    X_train, y_train, X_val, y_val = prepare_features(df, config.val_rounds)

    lgb_params = {
        "objective": "regression",
        "metric": "mae",
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_child_samples": config.min_child_samples,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "verbose": -1,
        "n_jobs": -1,
    }

    with mlflow.start_run():
        mlflow.log_params({**lgb_params, "training_seasons": config.training_seasons})

        model = lgb.LGBMRegressor(**lgb_params)  # type: ignore[arg-type]
        has_val = len(y_val) > 0
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)] if has_val else None,
            callbacks=(
                [
                    lgb.early_stopping(config.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(100),
                ]
                if has_val
                else [lgb.log_evaluation(100)]
            ),
        )

        if has_val:
            preds = model.predict(X_val)
            mae = mean_absolute_error(y_val, preds)
            r2 = r2_score(y_val, preds)
            mlflow.log_metrics({"val_mae_s": mae, "val_r2": r2})
            logger.info("Val  MAE=%.3f s  R²=%.3f", mae, r2)
        else:
            logger.warning("No validation laps (val rounds not yet ingested) — skipping metrics.")

        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=config.registered_model_name,
        )
        logger.info("Model registered as '%s'", config.registered_model_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    seasons: Annotated[
        str,
        typer.Option("--seasons", help="Comma-separated list of seasons, e.g. 2022,2023,2024."),
    ] = "2024",
) -> None:
    """CLI entry point for tire degradation model training."""
    season_list = [int(s.strip()) for s in seasons.split(",")]
    train(TireDegConfig(training_seasons=season_list))


if __name__ == "__main__":
    app()
