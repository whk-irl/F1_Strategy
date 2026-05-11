"""Unit tests for pandera schemas (bronze / silver / gold).

Validates that the schemas accept valid data and reject malformed data with
clear pandera ``SchemaError`` exceptions.
"""

from __future__ import annotations

import pandas as pd
import pandera as pa
import pytest
from services.ingestion.schemas.bronze import BronzeLapSchema
from services.ingestion.schemas.silver import SilverLapSchema


class TestBronzeLapSchema:
    def test_valid_dataframe_passes(self, bronze_laps_df: pd.DataFrame) -> None:
        BronzeLapSchema.validate(bronze_laps_df)

    def test_missing_required_column_raises(self, bronze_laps_df: pd.DataFrame) -> None:
        df = bronze_laps_df.drop(columns=["Driver"])
        with pytest.raises(pa.errors.SchemaError):
            BronzeLapSchema.validate(df)

    def test_negative_lap_number_raises(self, bronze_laps_df: pd.DataFrame) -> None:
        df = bronze_laps_df.copy()
        df.loc[0, "LapNumber"] = -1
        with pytest.raises(pa.errors.SchemaError):
            BronzeLapSchema.validate(df)

    def test_null_compound_allowed(self, bronze_laps_df: pd.DataFrame) -> None:
        df = bronze_laps_df.copy()
        df.loc[0, "Compound"] = None  # e.g. Monaco where compound data is missing
        BronzeLapSchema.validate(df)  # should not raise

    def test_null_lap_time_allowed(self, bronze_laps_df: pd.DataFrame) -> None:
        df = bronze_laps_df.copy()
        df.loc[0, "LapTime"] = pd.NaT
        BronzeLapSchema.validate(df)  # should not raise

    def test_extra_columns_tolerated(self, bronze_laps_df: pd.DataFrame) -> None:
        df = bronze_laps_df.copy()
        df["SomeExtraColumn"] = 42
        BronzeLapSchema.validate(df)  # strict=False


class TestSilverLapSchema:
    def test_valid_silver_passes(self, silver_laps_df: pd.DataFrame) -> None:
        SilverLapSchema.validate(silver_laps_df)

    def test_extra_column_raises(self, silver_laps_df: pd.DataFrame) -> None:
        df = silver_laps_df.copy()
        df["extra"] = 0
        with pytest.raises((pa.errors.SchemaError, pa.errors.SchemaErrors)):
            SilverLapSchema.validate(df)

    def test_invalid_session_raises(self, silver_laps_df: pd.DataFrame) -> None:
        df = silver_laps_df.copy()
        df["session"] = "INVALID"
        with pytest.raises(pa.errors.SchemaError):
            SilverLapSchema.validate(df)

    def test_unknown_compound_allowed(self, silver_laps_df: pd.DataFrame) -> None:
        df = silver_laps_df.copy()
        df["compound"] = "UNKNOWN"
        SilverLapSchema.validate(df)  # UNKNOWN is valid in silver

    def test_year_out_of_range_raises(self, silver_laps_df: pd.DataFrame) -> None:
        df = silver_laps_df.copy()
        df["year"] = 2010  # pre-FastF1 coverage
        with pytest.raises(pa.errors.SchemaError):
            SilverLapSchema.validate(df)
