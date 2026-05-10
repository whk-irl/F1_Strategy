"""Safety car model — inference.

Returns probability that a safety car or VSC will appear within the next
3 laps, given the current race state.
"""

from __future__ import annotations

import os

import mlflow.pyfunc
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

_MODEL_URI = os.getenv(
    "PITWALL_SC_MODEL_URI",
    "models:/pitwall-safety-car/latest",
)

# Threshold tuned for recall: better to pit unnecessarily than miss a free stop.
DEFAULT_THRESHOLD = 0.30


def load_model() -> mlflow.pyfunc.PyFuncModel:
    """Load the registered safety car model from MLflow."""
    tracking_uri = os.getenv("PITWALL_MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.pyfunc.load_model(_MODEL_URI)


def predict_sc_probability(
    race_progress: float,
    lap_number: int,
    sc_laps_since_last: float,
    compound_encoded: int,
    tyre_life_laps: float,
    position: float,
    model: mlflow.pyfunc.PyFuncModel | None = None,
) -> float:
    """Predict the probability of a safety car *starting* within the next 3 laps.

    This predicts SC onset from green-flag conditions only.  If an SC is already
    active (track_status_encoded >= 2) the caller already knows to pit — call
    this function only on green-flag laps.

    Args:
        race_progress: Fraction of race completed (0.0–1.0).
        lap_number: Current lap number.
        sc_laps_since_last: Laps since the last SC/VSC period (capped at 50).
        compound_encoded: Current tyre compound code.
        tyre_life_laps: Laps on current tyre set.
        position: Driver's current race position.
        model: Pre-loaded model (loads from MLflow registry if None).

    Returns:
        Probability in [0, 1] that a SC/VSC starts within the next 3 laps.
    """
    if model is None:
        model = load_model()

    row = pd.DataFrame(
        [
            {
                "race_progress": race_progress,
                "lap_number": lap_number,
                "sc_laps_since_last": min(sc_laps_since_last, 50.0),
                "compound_encoded": compound_encoded,
                "tyre_life_laps": tyre_life_laps,
                "position": position,
            }
        ]
    )
    proba = model.predict(row)
    # MLflow pyfunc returns probabilities for both classes; take positive class.
    return float(proba[0]) if proba.ndim == 1 else float(proba[0, 1])


def is_sc_likely(
    *,
    threshold: float = DEFAULT_THRESHOLD,
    **kwargs: object,
) -> bool:
    """Return True if safety car onset probability exceeds the decision threshold.

    Args:
        threshold: Probability cutoff (default 0.30 — recall-optimised).
        **kwargs: Forwarded to ``predict_sc_probability``.  Only call on
            green-flag laps (track_status_encoded == 0).

    Returns:
        True if SC onset is likely within the next 3 laps.
    """
    return predict_sc_probability(**kwargs) >= threshold  # type: ignore[arg-type]
