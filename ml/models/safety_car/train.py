"""Safety car probability model — LightGBM binary classification training.

Target: ``is_sc_next_3_laps`` — will a safety car or VSC appear within
        the next 3 laps? Predicting 3 laps ahead gives enough lead time for
        a pit-stop decision.

Class imbalance is handled via LightGBM's ``is_unbalance=True``.  Recall is
prioritised over precision (false negative = missed free pit far worse than
false positive = unnecessary pit).

Entry point:
    python -m ml.models.safety_car.train
    make train-safety-car
"""

from __future__ import annotations

import logging
from typing import Annotated

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import typer
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score

from ml.models._loader import load_gold_seasons

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

app = typer.Typer(add_completion=False)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [
    "race_progress",
    "lap_number",
    "sc_laps_since_last",
    "compound_encoded",
    "tyre_life_laps",
    "position",
]

TARGET_COL = "is_sc_next_3_laps"
SC_HORIZON = 3  # predict SC within this many laps


class SafetyCarConfig(BaseSettings):
    """Hyper-parameters and data settings for safety car model training."""

    model_config = SettingsConfigDict(env_prefix="PITWALL_SC_", env_file=".env", extra="ignore")

    training_seasons: list[int] = Field(default=[2024])
    val_rounds: list[int] = Field(default=[5, 10, 15, 20])

    n_estimators: int = Field(default=500)
    learning_rate: float = Field(default=0.05)
    num_leaves: int = Field(default=31)
    min_child_samples: int = Field(default=50)
    subsample: float = Field(default=0.8)
    colsample_bytree: float = Field(default=0.8)
    early_stopping_rounds: int = Field(default=50)

    mlflow_tracking_uri: str = Field(
        default="mlruns", validation_alias="PITWALL_MLFLOW_TRACKING_URI"
    )
    mlflow_experiment: str = Field(default="safety-car")
    registered_model_name: str = Field(default="pitwall-safety-car")


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------


def _build_sc_label(df: pd.DataFrame, horizon: int = SC_HORIZON) -> pd.DataFrame:
    """Add ``is_sc_next_N_laps`` label: True if SC/VSC appears within horizon laps.

    Groups by race (year + round_number) — the SC horizon is only within one race.

    Args:
        df: Gold DataFrame with ``track_status_encoded`` column.
        horizon: Look-ahead window in laps.

    Returns:
        DataFrame with the binary target column appended.
    """
    out = df.copy()
    # Collapse to one row per lap (pick any driver — SC affects whole race)
    race_status = (
        df.groupby(["year", "round_number", "lap_number"])["track_status_encoded"]
        .max()
        .reset_index()
        .sort_values(["year", "round_number", "lap_number"])
    )

    # For each lap, check if SC/VSC (encoded >= 2) appears in the next `horizon` laps
    race_status["is_sc_next"] = False
    for _grp_keys, grp in race_status.groupby(["year", "round_number"]):
        idx = grp.index
        sc_flags = (grp["track_status_encoded"] >= 2).values
        rolling = np.zeros(len(sc_flags), dtype=bool)
        for i in range(len(sc_flags)):
            rolling[i] = sc_flags[i : i + horizon].any()
        race_status.loc[idx, "is_sc_next"] = rolling

    out = out.merge(
        race_status[["year", "round_number", "lap_number", "is_sc_next"]],
        on=["year", "round_number", "lap_number"],
        how="left",
    )
    out[TARGET_COL] = out["is_sc_next"].fillna(False).astype(int)
    return out


def prepare_features(
    df: pd.DataFrame,
    val_rounds: list[int],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Build labels, filter, and split into train / validation sets.

    Args:
        df: Raw gold DataFrame.
        val_rounds: Round numbers held out for validation.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val).
    """
    labelled = _build_sc_label(df)

    mask = (
        (labelled["session"] == "R")
        # Only predict SC *onset* from green-flag laps.  Including laps where
        # track_status_encoded >= 2 causes leakage: the model trivially learns
        # "SC now → SC next lap" across multi-lap SC periods.
        & (labelled["track_status_encoded"] == 0)
        & labelled["lap_number"].notna()
        & labelled["position"].notna()
        & labelled["tyre_life_laps"].notna()
        & labelled["sc_laps_since_last"].notna()
    )
    clean = labelled[mask].copy()
    clean["sc_laps_since_last"] = clean["sc_laps_since_last"].clip(upper=50)

    # De-duplicate per (race, lap) — keep one row per lap since SC label is race-wide
    clean = clean.drop_duplicates(subset=["year", "round_number", "lap_number"])

    val_mask = clean["round_number"].isin(val_rounds)
    train = clean[~val_mask]
    val = clean[val_mask]

    sc_rate = clean[TARGET_COL].mean() * 100
    logger.info(
        "Dataset: %d train | %d val laps | SC label rate=%.1f%%",
        len(train),
        len(val),
        sc_rate,
    )

    if len(train) < 100:
        logger.warning(
            "Training set very small (%d rows). Ingest more seasons for a robust model.",
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


def train(config: SafetyCarConfig | None = None) -> None:
    """Train the safety car probability model and register it in MLflow.

    Args:
        config: Training configuration (defaults to env-var / defaults).
    """
    if config is None:
        config = SafetyCarConfig()

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment)

    df = load_gold_seasons(config.training_seasons)
    X_train, y_train, X_val, y_val = prepare_features(df, config.val_rounds)

    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_child_samples": config.min_child_samples,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "is_unbalance": True,  # handles rare SC events without manual resampling
        "verbose": -1,
        "n_jobs": -1,
    }

    with mlflow.start_run():
        mlflow.log_params({**lgb_params, "training_seasons": config.training_seasons})

        model = lgb.LGBMClassifier(**lgb_params)
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

        if has_val and y_val.sum() > 0:
            proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, proba)
            ap = average_precision_score(y_val, proba)
            mlflow.log_metrics({"val_roc_auc": auc, "val_avg_precision": ap})
            logger.info("Val  ROC-AUC=%.3f  Avg-Precision=%.3f", auc, ap)
            logger.info(
                "\n%s",
                classification_report(y_val, (proba >= 0.3).astype(int), zero_division=0),
            )
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
    """CLI entry point for safety car model training."""
    season_list = [int(s.strip()) for s in seasons.split(",")]
    train(SafetyCarConfig(training_seasons=season_list))


if __name__ == "__main__":
    app()
