"""Silver layer schema — cleaned, canonical per-lap records.

At this layer:
- Column names are snake_case.
- Timedeltas have been converted to float seconds.
- Compound is an enum-like string with no nulls (unknown → 'UNKNOWN').
- Every row has a race identifier (year, round_number, session).
- Inaccurate laps are retained but flagged; downstream models filter as needed.
"""

from __future__ import annotations

import pandera as pa
from pandera.typing import Series


class SilverLapSchema(pa.DataFrameModel):
    """Per-lap canonical record after cleaning."""

    # Race identifiers
    year: Series[int] = pa.Field(ge=2018, le=2030)
    round_number: Series[int] = pa.Field(ge=1, le=24)
    session: Series[str] = pa.Field(isin=["FP1", "FP2", "FP3", "Q", "SQ", "R", "S"])

    # Driver / team identifiers
    driver_code: Series[str] = pa.Field(
        str_length={"min_value": 2, "max_value": 3},
        description="Three-letter driver abbreviation.",
    )
    driver_number: Series[str] = pa.Field()
    team: Series[str] = pa.Field(description="Constructor name as reported by FastF1.")

    # Lap identifiers
    lap_number: Series[int] = pa.Field(ge=1)
    stint_number: Series[int] = pa.Field(ge=1)

    # Timing (seconds; NaN for in-laps, out-laps, and incomplete laps)
    lap_time_s: Series[float] = pa.Field(nullable=True, ge=0.0)
    sector1_s: Series[float] = pa.Field(nullable=True, ge=0.0)
    sector2_s: Series[float] = pa.Field(nullable=True, ge=0.0)
    sector3_s: Series[float] = pa.Field(nullable=True, ge=0.0)

    # Tire
    compound: Series[str] = pa.Field(
        isin=["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"],
    )
    tyre_life_laps: Series[float] = pa.Field(nullable=True, ge=0.0)
    is_fresh_tyre: Series[bool] = pa.Field(nullable=True)

    # Race state
    position: Series[float] = pa.Field(nullable=True, ge=1.0)
    track_status: Series[str] = pa.Field(nullable=True)
    pit_in_this_lap: Series[bool] = pa.Field(
        description="True if the car pitted at the end of this lap.",
    )
    pit_out_this_lap: Series[bool] = pa.Field(
        description="True if the car left the pits during this lap.",
    )
    is_accurate: Series[bool] = pa.Field()

    class Config:
        strict = True  # no extra columns — silver is the canonical contract
        coerce = True
