"""Bronze → Silver → Gold transforms for F1 lap data.

Each function is a pure transformation: it receives a DataFrame and returns a
new one.  Side-effects (storage, logging) belong in the pipeline layer.

Pandera validation is applied at the *output* of each transform so that
failures are caught close to where the bad data is produced.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd
import pandera as pa

from .schemas.bronze import BronzeLapSchema
from .schemas.gold import GoldLapSchema
from .schemas.silver import SilverLapSchema

logger = logging.getLogger(__name__)

# Maps FastF1 compound strings to integer labels for the gold layer.
COMPOUND_ENCODING: Final[dict[str, int]] = {
    "SOFT": 0,
    "MEDIUM": 1,
    "HARD": 2,
    "INTERMEDIATE": 3,
    "WET": 4,
    "UNKNOWN": 5,
}

# Track status string → integer for the gold layer.
TRACK_STATUS_ENCODING: Final[dict[str, int]] = {
    "1": 0,  # clear
    "2": 1,  # yellow flag
    "4": 2,  # VSC
    "6": 3,  # SC
    "5": 4,  # red flag
}


# ---------------------------------------------------------------------------
# Bronze → Silver
# ---------------------------------------------------------------------------


@pa.check_output(SilverLapSchema)
def bronze_to_silver(
    df: pd.DataFrame,
    year: int,
    round_number: int,
    session: str,
) -> pd.DataFrame:
    """Clean and canonicalize raw FastF1 laps into the silver schema.

    Args:
        df: Raw FastF1 laps DataFrame (validated by ``BronzeLapSchema`` upstream).
        year: Championship year.
        round_number: Round number within the season.
        session: Session type (e.g. ``'R'``).

    Returns:
        Silver DataFrame conforming to ``SilverLapSchema``.
    """
    out = df.copy()

    # --- Race identifiers ---
    out["year"] = year
    out["round_number"] = round_number
    out["session"] = session

    # --- Column renames ---
    out = out.rename(
        columns={
            "Driver": "driver_code",
            "DriverNumber": "driver_number",
            "LapNumber": "lap_number",
            "Stint": "stint_number",
            "Compound": "compound",
            "TyreLife": "tyre_life_laps",
            "FreshTyre": "is_fresh_tyre",
            "Position": "position",
            "TrackStatus": "track_status",
            "IsAccurate": "is_accurate",
        }
    )

    # --- Timedelta → seconds ---
    for src, dst in [
        ("LapTime", "lap_time_s"),
        ("Sector1Time", "sector1_s"),
        ("Sector2Time", "sector2_s"),
        ("Sector3Time", "sector3_s"),
    ]:
        if src in out.columns:
            out[dst] = _timedelta_to_seconds(out[src])
        else:
            out[dst] = np.nan

    # --- Pit flags ---
    # PitInTime is non-null when the car entered the pit lane on this lap.
    out["pit_in_this_lap"] = out.get("PitInTime", pd.Series(dtype=object)).notna()
    out["pit_out_this_lap"] = out.get("PitOutTime", pd.Series(dtype=object)).notna()

    # --- Compound normalisation ---
    out["compound"] = (
        out["compound"].fillna("UNKNOWN").str.upper().replace({"": "UNKNOWN"})
    )

    # --- Select and order final columns ---
    silver_cols = list(SilverLapSchema.to_schema().columns.keys())
    # Keep only columns present in both out and silver schema
    present = [c for c in silver_cols if c in out.columns]
    out = out[present]

    return out


# ---------------------------------------------------------------------------
# Silver → Gold
# ---------------------------------------------------------------------------


@pa.check_output(GoldLapSchema)
def silver_to_gold(df: pd.DataFrame, total_laps: int) -> pd.DataFrame:
    """Engineer ML features from the silver canonical lap table.

    Args:
        df: Silver DataFrame conforming to ``SilverLapSchema``.
        total_laps: Scheduled race distance in laps (used for ``race_progress``).

    Returns:
        Gold DataFrame conforming to ``GoldLapSchema``.
    """
    out = df.copy()

    # --- race_progress ---
    out["race_progress"] = (out["lap_number"] / total_laps).clip(0.0, 1.0)

    # --- Compound encoding ---
    out["compound_encoded"] = out["compound"].map(COMPOUND_ENCODING).fillna(5).astype(int)

    # --- track_status_encoded ---
    out["track_status_encoded"] = _encode_track_status(out["track_status"])

    # --- SC laps since last ---
    out["sc_laps_since_last"] = _laps_since_sc(out)

    # --- Per-driver, per-stint features ---
    group_driver_stint = out.groupby(["driver_code", "stint_number"], sort=False)

    out["lap_time_delta_s"] = group_driver_stint["lap_time_s"].transform(
        lambda s: s - s.median()
    )
    out["rolling_lap_time_3_s"] = group_driver_stint["lap_time_s"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    out["tyre_deg_rate_s_per_lap"] = group_driver_stint.apply(
        _compute_deg_rate, include_groups=False
    ).reset_index(level=[0, 1], drop=True)

    # --- Position change since stint start ---
    out["position_change_this_stint"] = group_driver_stint["position"].transform(
        lambda s: s - s.iloc[0]
    )

    return out


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _timedelta_to_seconds(series: pd.Series) -> pd.Series:  # type: ignore[type-arg]
    """Convert a pd.Timedelta series to float seconds (NaT → NaN)."""
    return series.dt.total_seconds()


def _encode_track_status(series: pd.Series) -> pd.Series:  # type: ignore[type-arg]
    """Map FastF1 track-status strings to a single integer priority code.

    FastF1 can return composite strings like ``"12"`` meaning yellow + clear.
    We take the *highest priority* code present.
    """
    priority = {"5": 4, "6": 3, "4": 2, "2": 1, "1": 0}

    def _map_row(val: object) -> int:
        if pd.isna(val):
            return 0
        chars = str(val)
        return max((priority.get(c, 0) for c in chars), default=0)

    return series.map(_map_row).astype(int)


def _laps_since_sc(df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
    """Return the number of laps elapsed since the last SC/VSC period ended."""
    # SC/VSC laps have track_status_encoded >= 2.
    encoded = _encode_track_status(df["track_status"])
    is_sc = encoded >= 2

    counter = np.zeros(len(df), dtype=float)
    c = np.nan
    for i, flag in enumerate(is_sc):
        if flag:
            c = 0.0
        elif not np.isnan(c):
            c += 1.0
        counter[i] = c

    return pd.Series(counter, index=df.index)


def _compute_deg_rate(group: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
    """OLS slope of lap_time_s vs tyre_life_laps within one stint for one driver.

    Returns NaN for stints with fewer than 3 valid laps (insufficient for a
    meaningful regression).
    """
    valid = group[["tyre_life_laps", "lap_time_s"]].dropna()
    if len(valid) < 3:  # noqa: PLR2004
        return pd.Series(np.nan, index=group.index)

    x = valid["tyre_life_laps"].values
    y = valid["lap_time_s"].values
    slope = float(np.polyfit(x, y, 1)[0])
    return pd.Series(max(slope, 0.0), index=group.index)
