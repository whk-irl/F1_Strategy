"""Data-quality contract tests for the bronze / silver / gold schemas.

These tests run as part of CI and are also designed to be run against real
Parquet files (integration mode) after a successful ingestion run.

Unit mode: uses synthetic fixtures from conftest.py.
Integration mode: reads from MinIO (requires ``make up``).
"""

from __future__ import annotations

import pandas as pd
import pytest
from services.ingestion.schemas.bronze import BronzeLapSchema
from services.ingestion.schemas.gold import GoldLapSchema
from services.ingestion.schemas.silver import SilverLapSchema
from services.ingestion.transforms import silver_to_gold


class TestBronzeContracts:
    """Contracts that must hold for any valid bronze Parquet."""

    def test_driver_codes_non_empty(self, bronze_laps_df: pd.DataFrame) -> None:
        assert not bronze_laps_df["Driver"].str.strip().eq("").any()

    def test_lap_numbers_positive(self, bronze_laps_df: pd.DataFrame) -> None:
        assert (bronze_laps_df["LapNumber"] >= 1).all()

    def test_stint_numbers_positive(self, bronze_laps_df: pd.DataFrame) -> None:
        assert (bronze_laps_df["Stint"] >= 1).all()

    def test_tyre_life_non_negative(self, bronze_laps_df: pd.DataFrame) -> None:
        valid = bronze_laps_df["TyreLife"].dropna()
        assert (valid >= 0).all()

    def test_position_in_valid_range(self, bronze_laps_df: pd.DataFrame) -> None:
        valid = bronze_laps_df["Position"].dropna()
        assert (valid >= 1).all()

    def test_schema_validates(self, bronze_laps_df: pd.DataFrame) -> None:
        BronzeLapSchema.validate(bronze_laps_df)


class TestSilverContracts:
    """Contracts that must hold after the bronze → silver transform."""

    def test_no_negative_lap_times(self, silver_laps_df: pd.DataFrame) -> None:
        valid = silver_laps_df["lap_time_s"].dropna()
        assert (valid > 0).all()

    def test_compound_values_valid(self, silver_laps_df: pd.DataFrame) -> None:
        allowed = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET", "UNKNOWN"}
        assert set(silver_laps_df["compound"].unique()).issubset(allowed)

    def test_no_null_compound(self, silver_laps_df: pd.DataFrame) -> None:
        assert silver_laps_df["compound"].notna().all()

    def test_pit_flags_are_bool(self, silver_laps_df: pd.DataFrame) -> None:
        assert silver_laps_df["pit_in_this_lap"].dtype == bool
        assert silver_laps_df["pit_out_this_lap"].dtype == bool

    def test_pit_in_and_out_mutually_exclusive_per_lap(self, silver_laps_df: pd.DataFrame) -> None:
        both_set = silver_laps_df["pit_in_this_lap"] & silver_laps_df["pit_out_this_lap"]
        assert not both_set.any(), "A lap cannot simultaneously be a pit-in and pit-out lap"

    def test_race_identifiers_consistent(self, silver_laps_df: pd.DataFrame) -> None:
        assert silver_laps_df["year"].nunique() == 1
        assert silver_laps_df["round_number"].nunique() == 1
        assert silver_laps_df["session"].nunique() == 1

    def test_schema_validates(self, silver_laps_df: pd.DataFrame) -> None:
        SilverLapSchema.validate(silver_laps_df)


class TestGoldContracts:
    """Contracts that must hold after the silver → gold transform."""

    @pytest.fixture
    def gold_laps_df(self, silver_laps_df: pd.DataFrame) -> pd.DataFrame:
        return silver_to_gold(silver_laps_df, total_laps=57)

    def test_race_progress_between_0_and_1(self, gold_laps_df: pd.DataFrame) -> None:
        assert gold_laps_df["race_progress"].between(0.0, 1.0).all()

    def test_no_negative_deg_rate(self, gold_laps_df: pd.DataFrame) -> None:
        valid = gold_laps_df["tyre_deg_rate_s_per_lap"].dropna()
        assert (valid >= 0.0).all()

    def test_compound_encoded_in_range(self, gold_laps_df: pd.DataFrame) -> None:
        assert gold_laps_df["compound_encoded"].between(0, 5).all()

    def test_track_status_encoded_in_range(self, gold_laps_df: pd.DataFrame) -> None:
        assert gold_laps_df["track_status_encoded"].between(0, 4).all()

    def test_sc_laps_since_last_non_negative(self, gold_laps_df: pd.DataFrame) -> None:
        valid = gold_laps_df["sc_laps_since_last"].dropna()
        assert (valid >= 0.0).all()

    def test_schema_validates(self, gold_laps_df: pd.DataFrame) -> None:
        GoldLapSchema.validate(gold_laps_df)
