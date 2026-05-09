"""Bronze layer schema — raw FastF1 lap data.

Minimal enforcement: we preserve the original shape from FastF1 and only
reject rows that are obviously malformed (e.g. impossible lap numbers).
All type coercion and column renaming happens in the silver transform.
"""

from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera.typing import Series


class BronzeLapSchema(pa.DataFrameModel):
    """Schema for the raw FastF1 ``session.laps`` DataFrame.

    ``strict=False`` allows extra columns that FastF1 may add across versions;
    ``coerce=True`` handles minor dtype mismatches (e.g. int64 vs int32).
    """

    # Identifiers
    Driver: Series[str] = pa.Field(
        description="Three-letter driver code (e.g. 'VER').",
    )
    DriverNumber: Series[str] = pa.Field(
        description="Car number as a string (FastF1 convention).",
    )
    LapNumber: Series[int] = pa.Field(ge=1, description="1-based lap counter.")
    Stint: Series[int] = pa.Field(ge=1, description="1-based stint counter.")

    # Timing — timedeltas; NaT for in/out laps and laps with no crossing.
    LapTime: Series[pd.Timedelta] = pa.Field(nullable=True)
    Sector1Time: Series[pd.Timedelta] = pa.Field(nullable=True)
    Sector2Time: Series[pd.Timedelta] = pa.Field(nullable=True)
    Sector3Time: Series[pd.Timedelta] = pa.Field(nullable=True)

    # Tire
    Compound: Series[str] = pa.Field(
        nullable=True,
        isin=["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"],
        description="Tire compound code from FastF1.",
    )
    TyreLife: Series[float] = pa.Field(nullable=True, ge=0)
    FreshTyre: Series[bool] = pa.Field(nullable=True)

    # Race state
    Position: Series[float] = pa.Field(nullable=True, ge=1)
    TrackStatus: Series[str] = pa.Field(
        nullable=True,
        description="Comma-joined status codes from the timing feed.",
    )
    IsAccurate: Series[bool] = pa.Field(
        description="FastF1 flag — False for laps with missing timing data.",
    )

    class Config:
        strict = False  # tolerate extra FastF1 columns
        coerce = True
