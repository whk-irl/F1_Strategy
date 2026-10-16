"""Bronze → Silver → Gold transforms for F1 lap data.

Each function is a pure transformation: it receives a DataFrame and returns a
new one.  Side-effects (storage, logging) belong in the pipeline layer.

Pandera validation is applied at the *output* of each transform so that
failures are caught close to where the bad data is produced.
"""

from __future__ import annotations

import logging
from typing import Any, Final, cast

import numpy as np
import pandas as pd
import pandera as pa

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
    "4": 2,  # SC (legacy numeric encoding retained for saved models)
    "6": 3,  # VSC
    "7": 3,  # VSC ending
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
            "Team": "team",
        }
    )
    if "team" not in out.columns:
        out["team"] = "UNKNOWN"

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
    out["pit_in_this_lap"] = out.get("PitInTime", pd.Series(index=out.index, dtype=object)).notna()
    out["pit_out_this_lap"] = out.get(
        "PitOutTime", pd.Series(index=out.index, dtype=object)
    ).notna()
    out["is_accurate"] = out["is_accurate"].fillna(False).astype(bool)
    out["lap_end_time_s"] = _timedelta_to_seconds(out["Time"]) if "Time" in out else np.nan
    for col in ("track_temp_c", "air_temp_c", "rainfall"):
        if col not in out:
            out[col] = np.nan

    # --- Compound normalisation ---
    out["compound"] = (
        out["compound"]
        .fillna("UNKNOWN")
        .str.upper()
        .replace({"": "UNKNOWN", "NONE": "UNKNOWN", "NAN": "UNKNOWN"})
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
def silver_to_gold(
    df: pd.DataFrame, total_laps: int, session_start: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Engineer ML features from the silver canonical lap table.

    Args:
        df: Silver DataFrame conforming to ``SilverLapSchema``.
        total_laps: Scheduled race distance in laps (used for ``race_progress``).

    Returns:
        Gold DataFrame conforming to ``GoldLapSchema``.
    """
    if total_laps < 1:
        raise ValueError("total_laps must be positive")
    if df.empty:
        raise ValueError("Cannot build gold features from an empty session")
    if len(df[["year", "round_number", "session"]].drop_duplicates()) != 1:
        raise ValueError("silver_to_gold expects exactly one session")
    out = df.sort_values(["driver_number", "stint_number", "lap_number"]).copy()

    # --- Feast event_timestamp — synthetic race date (year + round → approx date) ---
    year_val = int(out["year"].iloc[0])
    round_val = int(out["round_number"].iloc[0])
    race_date = pd.Timestamp(year=year_val, month=3, day=1, tz="UTC") + pd.Timedelta(
        weeks=(round_val - 1) * 2
    )
    # Legacy imports lack real timing; acquisition always supplies the actual UTC origin.
    origin = session_start if session_start is not None else race_date
    origin = pd.Timestamp(origin)
    origin = origin.tz_localize("UTC") if origin.tzinfo is None else origin.tz_convert("UTC")
    elapsed = out.get("lap_end_time_s", pd.Series(0.0, index=out.index)).fillna(0.0)
    out["event_timestamp"] = origin + pd.to_timedelta(elapsed, unit="s")

    # --- race_progress ---
    out["race_progress"] = (out["lap_number"] / total_laps).clip(0.0, 1.0)

    # --- Compound encoding ---
    out["compound_encoded"] = out["compound"].map(COMPOUND_ENCODING).fillna(5).astype(int)

    # --- track_status_encoded ---
    out["track_status_encoded"] = _encode_track_status(out["track_status"])

    # --- SC laps since last ---
    out["sc_laps_since_last"] = _laps_since_sc(out)

    # --- Per-driver, per-stint features ---
    group_driver_stint = out.groupby(["driver_number", "stint_number"], sort=False)

    clean_mask = (
        out["is_accurate"].fillna(False)
        & ~out["pit_in_this_lap"]
        & ~out["pit_out_this_lap"]
        & out["track_status_encoded"].eq(0)
    )
    out["_clean_lap"] = out["lap_time_s"].where(clean_mask)
    clean_groups = out.groupby(["driver_number", "stint_number"], sort=False)["_clean_lap"]
    out["lap_time_delta_s"] = out["_clean_lap"] - clean_groups.transform(
        lambda s: s.expanding(min_periods=1).median().shift(1)
    )
    out["rolling_lap_time_3_s"] = clean_groups.transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    out["tyre_deg_rate_s_per_lap"] = np.nan
    for _, group in group_driver_stint:
        out.loc[group.index, "tyre_deg_rate_s_per_lap"] = _compute_deg_rate(group)

    # --- Position change since stint start ---
    out["position_change_this_stint"] = group_driver_stint["position"].transform(
        lambda s: s - s.iloc[0]
    )

    # --- Team relative pace: driver lap time vs field median for that lap ---
    field_median = out.groupby("lap_number")["_clean_lap"].transform("median")
    out["lap_delta_to_field_median_s"] = out["lap_time_s"] - field_median

    return out.drop(columns=["_clean_lap"])


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _timedelta_to_seconds(series: pd.Series[Any]) -> pd.Series[Any]:
    """Convert a pd.Timedelta series to float seconds (NaT → NaN)."""
    return series.dt.total_seconds()  # type: ignore[no-any-return]


def _encode_track_status(series: pd.Series[Any]) -> pd.Series[Any]:
    """Map FastF1 track-status strings to a single integer priority code.

    FastF1 can return composite strings like ``"12"`` meaning yellow + clear.
    We take the *highest priority* code present.
    """
    priority = {"5": 4, "7": 3, "6": 3, "4": 2, "2": 1, "1": 0}

    def _map_row(val: object) -> int:
        if pd.isna(val):  # type: ignore[call-overload]
            return 0
        chars = str(val)
        return max((priority.get(c, 0) for c in chars), default=0)

    return series.map(_map_row).astype(int)


def _laps_since_sc(df: pd.DataFrame) -> pd.Series[Any]:
    """Return the number of laps elapsed since the last SC/VSC period ended."""
    encoded = _encode_track_status(df["track_status"])
    if "lap_number" in df:
        by_lap = encoded.groupby(df["lap_number"]).max().sort_index()
        last_sc = np.nan
        elapsed: dict[int, float] = {}
        for lap, status in by_lap.items():
            lap_number = int(cast(Any, lap))
            if status in (2, 3):
                last_sc = float(lap_number)
            elapsed[lap_number] = float(lap_number - last_sc)
        return df["lap_number"].map(elapsed)
    is_sc = encoded.isin([2, 3])

    counter = np.zeros(len(df), dtype=float)
    c = np.nan
    for i, flag in enumerate(is_sc):
        if flag:
            c = 0.0
        elif not np.isnan(c):
            c += 1.0
        counter[i] = c

    return pd.Series(counter, index=df.index)


def _compute_deg_rate(group: pd.DataFrame) -> pd.Series[Any]:
    """OLS slope of lap_time_s vs tyre_life_laps within one stint for one driver.

    Returns NaN for stints with fewer than 3 valid laps (insufficient for a
    meaningful regression).
    """
    valid = group["tyre_life_laps"].notna() & group["lap_time_s"].notna()
    if "_clean_lap" in group:
        valid &= group["_clean_lap"].notna()
    x = group["tyre_life_laps"].where(valid, 0.0).fillna(0.0)
    y = group["lap_time_s"].where(valid, 0.0).fillna(0.0)
    n = valid.astype(int).cumsum()
    denominator = n * (x * x).cumsum() - x.cumsum() ** 2
    numerator = n * (x * y).cumsum() - x.cumsum() * y.cumsum()
    return (numerator / denominator.where(denominator > 0)).where(n >= 3).clip(lower=0.0)
