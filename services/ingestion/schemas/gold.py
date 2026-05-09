"""Gold layer schema — ML-ready feature table.

Extends the silver schema with engineered features used by the tire
degradation, safety-car, and baseline pace models.  All values are float or
int; categoricals are one-hot encoded or label-encoded upstream.
"""

from __future__ import annotations

import pandera as pa
from pandera.typing import Series

from .silver import SilverLapSchema


class GoldLapSchema(SilverLapSchema):
    """Silver schema plus derived ML features."""

    # Pace features
    lap_time_delta_s: Series[float] = pa.Field(
        nullable=True,
        description="lap_time_s minus driver's median lap time in this stint.",
    )
    rolling_lap_time_3_s: Series[float] = pa.Field(
        nullable=True,
        description="Rolling 3-lap mean lap time for this driver.",
    )

    # Tire degradation proxy
    tyre_deg_rate_s_per_lap: Series[float] = pa.Field(
        nullable=True,
        ge=0.0,
        description="OLS slope of lap_time_s vs tyre_life_laps within this stint.",
    )

    # Safety-car context
    sc_laps_since_last: Series[float] = pa.Field(
        nullable=True,
        ge=0.0,
        description="Laps elapsed since the last safety car or VSC period ended.",
    )
    track_status_encoded: Series[int] = pa.Field(
        ge=0,
        description="Numeric encoding: 0=clear, 1=yellow, 2=VSC, 3=SC, 4=red.",
    )

    # Race progress
    race_progress: Series[float] = pa.Field(
        ge=0.0,
        le=1.0,
        description="lap_number / total_laps — normalised race progress.",
    )
    position_change_this_stint: Series[float] = pa.Field(
        nullable=True,
        description="Positions gained (+) or lost (-) since stint start.",
    )

    # Compound encoding (integer label, for models that prefer it)
    compound_encoded: Series[int] = pa.Field(
        ge=0,
        description="SOFT=0, MEDIUM=1, HARD=2, INTERMEDIATE=3, WET=4, UNKNOWN=5.",
    )

    class Config:
        strict = True
        coerce = True
