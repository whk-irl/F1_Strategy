"""Tire degradation model — inference.

Loads the latest registered MLflow model and predicts lap time delta
given tyre state inputs.
"""

from __future__ import annotations

import os

import mlflow.lightgbm
import numpy as np
import pandas as pd

_MODEL_URI = os.getenv(
    "PITWALL_TIRE_MODEL_URI",
    "models:/pitwall-tire-degradation/latest",
)


def load_model() -> mlflow.pyfunc.PyFuncModel:
    """Load the registered tire degradation model from MLflow."""
    tracking_uri = os.getenv("PITWALL_MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.pyfunc.load_model(_MODEL_URI)


def predict_lap_time_delta(
    tyre_life_laps: int,
    compound_encoded: int,
    race_progress: float,
    track_status_encoded: int,
    is_fresh_tyre: bool,
    lap_delta_to_field_median_s: float,
    model: mlflow.pyfunc.PyFuncModel | None = None,
) -> float:
    """Predict the lap time delta (seconds vs driver median) for given tyre state.

    Args:
        tyre_life_laps: Number of laps on the current set of tyres.
        compound_encoded: 0=SOFT, 1=MEDIUM, 2=HARD, 3=INTER, 4=WET, 5=UNKNOWN.
        race_progress: Fraction of race completed (0.0–1.0).
        track_status_encoded: 0=clear, 1=yellow, 2=VSC, 3=SC, 4=red.
        is_fresh_tyre: Whether the tyre set is new.
        lap_delta_to_field_median_s: Car's typical pace vs field (seconds).
        model: Pre-loaded model (loads from MLflow registry if None).

    Returns:
        Predicted lap time delta in seconds relative to the driver's stint median.
    """
    if model is None:
        model = load_model()

    row = pd.DataFrame(
        [
            {
                "tyre_life_laps": tyre_life_laps,
                "compound_encoded": compound_encoded,
                "race_progress": race_progress,
                "track_status_encoded": track_status_encoded,
                "is_fresh_tyre": int(is_fresh_tyre),
                "lap_delta_to_field_median_s": lap_delta_to_field_median_s,
            }
        ]
    )
    return float(model.predict(row)[0])


def predict_stint_degradation(
    stint_laps: int,
    compound_encoded: int,
    race_progress_start: float,
    lap_delta_to_field_median_s: float,
    model: mlflow.pyfunc.PyFuncModel | None = None,
) -> np.ndarray:
    """Predict lap time deltas for an entire projected stint.

    Args:
        stint_laps: Number of laps to project forward.
        compound_encoded: Compound integer code.
        race_progress_start: Race progress at stint start (0.0–1.0).
        lap_delta_to_field_median_s: Car baseline pace delta.
        model: Pre-loaded model (loads if None).

    Returns:
        Array of predicted lap time deltas, one per lap in the stint.
    """
    if model is None:
        model = load_model()

    rows = pd.DataFrame(
        {
            "tyre_life_laps": np.arange(1, stint_laps + 1),
            "compound_encoded": compound_encoded,
            "race_progress": np.linspace(
                race_progress_start,
                min(race_progress_start + stint_laps * 0.02, 1.0),
                stint_laps,
            ),
            "track_status_encoded": 0,
            "is_fresh_tyre": [1] + [0] * (stint_laps - 1),
            "lap_delta_to_field_median_s": lap_delta_to_field_median_s,
        }
    )
    return model.predict(rows)
