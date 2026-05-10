"""Feast feature store definitions — scanned by ``feast apply``.

This file is intentionally self-contained (no relative imports) because
Feast imports it as a top-level module, not as part of the ml.features
package.  The modular files (entities.py, data_sources.py, feature_views.py)
expose the same objects for programmatic use by training scripts.
"""

from __future__ import annotations

import os
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.data_format import ParquetFormat
from feast.types import Bool, Float32, Int32, String

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------

_backend = os.getenv("PITWALL_STORAGE_BACKEND", "s3")
_bucket = os.getenv("PITWALL_AWS_S3_BUCKET", "pitwall-ai-prod")
_minio_bucket = os.getenv("PITWALL_MINIO_BUCKET", "pitwall-ai")
_gold_path = f"s3://{_bucket}/gold/" if _backend == "s3" else f"s3://{_minio_bucket}/gold/"

gold_laps_source = FileSource(
    name="gold_laps",
    path=_gold_path,
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    description="Gold-layer per-lap feature table, Hive-partitioned on S3.",
)

# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

driver = Entity(
    name="driver",
    join_keys=["driver_number"],
    value_type=ValueType.STRING,
    description="F1 driver identified by their race number (e.g. '1', '44').",
)

# ---------------------------------------------------------------------------
# Feature views
# ---------------------------------------------------------------------------

_TTL = timedelta(days=3650)

lap_tire_fv = FeatureView(
    name="lap_tire_fv",
    entities=[driver],
    ttl=_TTL,
    schema=[
        Field(name="compound", dtype=String),
        Field(name="compound_encoded", dtype=Int32),
        Field(name="tyre_life_laps", dtype=Float32),
        Field(name="is_fresh_tyre", dtype=Bool),
        Field(name="tyre_deg_rate_s_per_lap", dtype=Float32),
        Field(name="pit_in_this_lap", dtype=Bool),
        Field(name="pit_out_this_lap", dtype=Bool),
    ],
    source=gold_laps_source,
    description="Tyre state features per driver per lap.",
)

lap_race_context_fv = FeatureView(
    name="lap_race_context_fv",
    entities=[driver],
    ttl=_TTL,
    schema=[
        Field(name="race_progress", dtype=Float32),
        Field(name="track_status_encoded", dtype=Int32),
        Field(name="sc_laps_since_last", dtype=Float32),
        Field(name="position", dtype=Float32),
        Field(name="position_change_this_stint", dtype=Float32),
        Field(name="stint_number", dtype=Int32),
        Field(name="lap_number", dtype=Int32),
        Field(name="round_number", dtype=Int32),
        Field(name="year", dtype=Int32),
        Field(name="session", dtype=String),
    ],
    source=gold_laps_source,
    description="Race situation features per driver per lap.",
)

lap_pace_fv = FeatureView(
    name="lap_pace_fv",
    entities=[driver],
    ttl=_TTL,
    schema=[
        Field(name="lap_time_s", dtype=Float32),
        Field(name="lap_time_delta_s", dtype=Float32),
        Field(name="rolling_lap_time_3_s", dtype=Float32),
        Field(name="lap_delta_to_field_median_s", dtype=Float32),
        Field(name="team", dtype=String),
        Field(name="sector1_s", dtype=Float32),
        Field(name="sector2_s", dtype=Float32),
        Field(name="sector3_s", dtype=Float32),
    ],
    source=gold_laps_source,
    description=(
        "Pace and relative performance features. "
        "lap_delta_to_field_median_s captures within-race car pace vs the field."
    ),
)
