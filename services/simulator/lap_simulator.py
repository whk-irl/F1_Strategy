"""Core lap-time simulation helpers shared by F1RaceEnv and the validator.

Given a driver's tyre state and race context, a simulated lap time is built from
three additive components:

    lap_time = field_median[lap] + driver_baseline_delta + tire_model_delta

where:
  - ``field_median[lap]`` is the median of all drivers' actual lap times for
    that lap (a circuit-pace reference that varies as the race evolves).
  - ``driver_baseline_delta`` is the driver/car's typical pace vs the field
    median, estimated from the first 10 clean green-flag laps.
  - ``tire_model_delta`` is the LightGBM tire-degradation model output: the
    predicted deviation from the driver's own stint median, driven by compound
    and tyre age.

Pit-in laps add a configurable ``pit_loss_s`` penalty (default 25 s).
"""

from __future__ import annotations

import logging

import pandas as pd
from ml.models.tire_degradation.predict import predict_lap_time_delta

logger = logging.getLogger(__name__)

PIT_LOSS_S: float = 25.0  # typical pit-lane delta at most circuits


# ---------------------------------------------------------------------------
# Race-level helpers
# ---------------------------------------------------------------------------


def field_median_by_lap(race_df: pd.DataFrame) -> pd.Series:
    """Median lap time across all drivers for each lap (green-flag, non-pit only).

    Args:
        race_df: Gold DataFrame for a single race, all drivers.

    Returns:
        Series indexed by ``lap_number`` → median lap time in seconds.
    """
    clean = race_df[
        race_df["is_accurate"].astype(bool)
        & (race_df["track_status_encoded"] == 0)
        & (~race_df["pit_in_this_lap"].astype(bool))
        & (~race_df["pit_out_this_lap"].astype(bool))
        & race_df["lap_time_s"].notna()
    ]
    return clean.groupby("lap_number")["lap_time_s"].median()


def driver_baseline_delta(driver_df: pd.DataFrame) -> float:
    """Estimate driver/car baseline pace delta vs field median.

    Uses the first 10 clean green-flag laps to isolate car speed from tyre
    degradation.

    Args:
        driver_df: Gold rows for one driver in one race.

    Returns:
        Mean ``lap_delta_to_field_median_s`` over early green laps,
        or 0.0 if no clean laps are found.
    """
    early = driver_df[
        (driver_df["lap_number"] <= 10)
        & driver_df["lap_delta_to_field_median_s"].notna()
        & (driver_df["track_status_encoded"] == 0)
        & (~driver_df["pit_in_this_lap"].astype(bool))
        & (~driver_df["pit_out_this_lap"].astype(bool))
    ]
    if early.empty:
        return 0.0
    return float(early["lap_delta_to_field_median_s"].mean())


# ---------------------------------------------------------------------------
# Driver-race replay
# ---------------------------------------------------------------------------


def simulate_driver_race(
    driver_df: pd.DataFrame,
    field_medians: pd.Series,
    tire_model: object,
    pit_loss_s: float = PIT_LOSS_S,
) -> pd.DataFrame:
    """Replay a driver's race using their actual pit decisions and the tire model.

    Args:
        driver_df: Gold rows for one driver, one race (any order).
        field_medians: Output of :func:`field_median_by_lap`.
        tire_model: Loaded MLflow pyfunc tire-degradation model.
        pit_loss_s: Seconds added on each pit-in lap.

    Returns:
        DataFrame with columns ``lap_number`` and ``simulated_lap_time_s``.
    """
    rows = driver_df.sort_values("lap_number").reset_index(drop=True)
    baseline = driver_baseline_delta(rows)
    total_laps = int(rows["lap_number"].max()) if not rows.empty else 1

    records: list[dict[str, float]] = []
    for _, row in rows.iterrows():
        lap = int(row["lap_number"])
        compound = int(row["compound_encoded"]) if pd.notna(row["compound_encoded"]) else 1
        tyre_age = max(1, int(row["tyre_life_laps"])) if pd.notna(row["tyre_life_laps"]) else 1
        is_fresh = bool(row.get("is_fresh_tyre", False))
        race_prog = float(row.get("race_progress", lap / total_laps))

        # Field median for this lap with nearest-neighbour fallback
        field_med: float | None = field_medians.get(lap)
        if field_med is None or pd.isna(field_med):
            neighbours = [
                field_medians.get(lap + d)
                for d in (-1, 1, -2, 2, -3, 3)
                if field_medians.get(lap + d) is not None
            ]
            field_med = float(neighbours[0]) if neighbours else 90.0

        tire_delta = predict_lap_time_delta(
            tyre_life_laps=tyre_age,
            compound_encoded=compound,
            race_progress=race_prog,
            track_status_encoded=0,  # model trained on green laps; SC laps are noise
            is_fresh_tyre=is_fresh,
            lap_delta_to_field_median_s=baseline,
            model=tire_model,
        )

        sim_time = float(field_med) + baseline + tire_delta
        if bool(row.get("pit_in_this_lap", False)):
            sim_time += pit_loss_s

        records.append({"lap_number": float(lap), "simulated_lap_time_s": sim_time})

    return pd.DataFrame(records)
