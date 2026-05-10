"""Tyre degradation calibration from practice long-run data.

Uses FP1/FP2 lap data already in the gold layer to fit per-compound
degradation rates for a specific race weekend.  The fitted rates override
the circuit table defaults in :mod:`services.simulator.env`, making the
simulation more accurate for that round.

Typical use:
    from services.simulator.tyre_calibration import calibrate
    rates = calibrate(gold_df, round_num=16, season=2024)  # Italian GP
    env = F1RaceEnv(race_df, driver, tire_model, sc_model,
                    calibrated_rate=rates["degrade_rate"],
                    calibrated_cliff=rates["cliff_lap"])
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.linear_model import HuberRegressor

logger = logging.getLogger(__name__)

# Minimum laps in a clean stint to use it for fitting.
_MIN_STINT_LAPS = 8

# compound_encoded → human name (for logging)
_COMPOUND_NAME: dict[int, str] = {0: "SOFT", 1: "MEDIUM", 2: "HARD"}

# Fallback rates if a compound has no usable practice data.
_FALLBACK_RATE: dict[int, float] = {0: 0.08, 1: 0.05, 2: 0.03}
_FALLBACK_CLIFF: dict[int, int] = {0: 22, 1: 32, 2: 42}


def calibrate(
    gold_df: pd.DataFrame,
    round_num: int,
    season: int,
    sessions: list[str] | None = None,
) -> dict[str, dict[int, float] | dict[int, int]]:
    """Fit per-compound degradation rates from practice long runs.

    Searches for FP1/FP2 data in ``gold_df`` for the given round and season.
    Finds clean long stints (≥8 laps, no SC, no pit-in/out), fits a Huber
    regression of lap_time_delta_s ~ tyre_life_laps per compound, and
    estimates the cliff lap from where the marginal degradation doubles.

    Args:
        gold_df: Full gold DataFrame (all sessions, all rounds).
        round_num: Round number to calibrate for (e.g. 16 for Monza).
        season: Championship year (e.g. 2024).
        sessions: Practice session names to use.  Defaults to ["FP2", "FP1"]
                  (FP2 long runs are more representative of race conditions).

    Returns:
        Dict with keys:
            ``"degrade_rate"`` — compound_encoded → fitted rate (s/lap)
            ``"cliff_lap"``    — compound_encoded → estimated cliff lap
        Falls back to circuit table defaults for compounds with insufficient data.
    """
    if sessions is None:
        sessions = ["FP2", "FP1"]

    practice = gold_df[
        (gold_df["year"] == season)
        & (gold_df["round_number"] == round_num)
        & gold_df["session"].isin(sessions)
    ].copy()

    if practice.empty:
        logger.warning(
            "No practice data for round %d / %d — using circuit defaults.", round_num, season
        )
        return {"degrade_rate": {}, "cliff_lap": {}}

    fitted_rate: dict[int, float] = {}
    fitted_cliff: dict[int, int] = {}

    for compound in [0, 1, 2]:  # SOFT, MEDIUM, HARD only (not INTER/WET)
        stints = _extract_clean_stints(practice, compound)
        if stints.empty:
            logger.info(
                "Round %d %s: no clean stints — using default.",
                round_num,
                _COMPOUND_NAME[compound],
            )
            continue

        rate, cliff = _fit_degradation(stints, compound)
        if rate is not None:
            fitted_rate[compound] = rate
            fitted_cliff[compound] = cliff
            logger.info(
                "Round %d %s: rate=%.4f s/lap, cliff≈lap %d  (n=%d laps)",
                round_num,
                _COMPOUND_NAME[compound],
                rate,
                cliff,
                len(stints),
            )

    return {"degrade_rate": fitted_rate, "cliff_lap": fitted_cliff}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_clean_stints(practice_df: pd.DataFrame, compound: int) -> pd.DataFrame:
    """Return clean long-run laps for a given compound.

    Filters out: pit laps, SC/VSC laps, inaccurate laps, first out-lap of stint.
    Groups by driver × stint and keeps only stints with ≥ _MIN_STINT_LAPS laps.
    """
    required = {"lap_time_s", "tyre_life_laps", "compound_encoded"}
    if not required.issubset(practice_df.columns):
        return pd.DataFrame()

    mask = (
        (practice_df["compound_encoded"] == compound)
        & practice_df["lap_time_s"].notna()
        & practice_df["tyre_life_laps"].notna()
        & (practice_df["tyre_life_laps"] >= 2)  # skip out-lap warm-up
    )
    if "is_accurate" in practice_df.columns:
        mask &= practice_df["is_accurate"].astype(bool)
    if "track_status_encoded" in practice_df.columns:
        mask &= practice_df["track_status_encoded"] == 0
    if "pit_in_this_lap" in practice_df.columns:
        mask &= ~practice_df["pit_in_this_lap"].astype(bool)
    if "pit_out_this_lap" in practice_df.columns:
        mask &= ~practice_df["pit_out_this_lap"].astype(bool)

    clean = practice_df[mask].copy()
    if clean.empty:
        return clean

    # Keep only stints long enough to fit a meaningful trend.
    clean["stint_id"] = clean.groupby("driver_number")["tyre_life_laps"].transform(
        lambda s: (s.diff().fillna(1) < 0).cumsum()
    )
    stint_lengths = clean.groupby(["driver_number", "stint_id"])["lap_time_s"].transform("count")
    return clean[stint_lengths >= _MIN_STINT_LAPS].reset_index(drop=True)


def _fit_degradation(
    stints: pd.DataFrame,
    compound: int,
) -> tuple[float, int] | tuple[None, None]:
    """Fit Huber regression to estimate degradation rate and cliff lap.

    Returns:
        (rate_s_per_lap, cliff_lap) or (None, None) if fit fails.
    """
    X = stints["tyre_life_laps"].values.reshape(-1, 1)
    y = stints["lap_time_s"].values

    if len(X) < _MIN_STINT_LAPS:
        return None, None

    try:
        model = HuberRegressor(epsilon=1.35, max_iter=200)
        model.fit(X, y)
        rate = float(model.coef_[0])

        if rate <= 0:
            # Negative slope means the data is too noisy for this compound.
            return None, None

        # Estimate cliff: lap where degradation rate doubles (post-cliff rate = 4× linear).
        # For a linear fit, we approximate cliff as the lap where residuals start to
        # grow above 1.5× the linear prediction — simpler: use compound defaults
        # scaled by measured vs expected rate.
        default_rate = _FALLBACK_RATE.get(compound, 0.05)
        default_cliff = _FALLBACK_CLIFF.get(compound, 35)
        rate_ratio = rate / default_rate
        # Higher degradation → earlier cliff; lower → later cliff.
        cliff = max(5, int(default_cliff / max(rate_ratio, 0.1)))

        return rate, cliff

    except Exception as exc:
        logger.debug("Degradation fit failed for compound %d: %s", compound, exc)
        return None, None
