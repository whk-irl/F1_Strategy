"""Unit tests for services/ingestion/transforms.py.

These tests use the synthetic fixtures from conftest.py and do not require
any external services (no MinIO, no MLflow, no FastF1 network calls).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.ingestion.transforms import (
    COMPOUND_ENCODING,
    _encode_track_status,
    _laps_since_sc,
    bronze_to_silver,
    silver_to_gold,
)


class TestBronzeToSilver:
    def test_output_columns_match_schema(self, bronze_laps_df: pd.DataFrame) -> None:
        from services.ingestion.schemas.silver import SilverLapSchema

        silver = bronze_to_silver(bronze_laps_df, year=2024, round_number=1, session="R")
        expected = set(SilverLapSchema.to_schema().columns.keys())
        assert expected == set(silver.columns)

    def test_race_identifiers_set(self, bronze_laps_df: pd.DataFrame) -> None:
        silver = bronze_to_silver(bronze_laps_df, year=2024, round_number=3, session="Q")
        assert (silver["year"] == 2024).all()
        assert (silver["round_number"] == 3).all()
        assert (silver["session"] == "Q").all()

    def test_lap_time_converted_to_seconds(self, bronze_laps_df: pd.DataFrame) -> None:
        silver = bronze_to_silver(bronze_laps_df, year=2024, round_number=1, session="R")
        # Lap 1 raw LapTime is 90.5 s
        lap1 = silver[silver["lap_number"] == 1]
        assert abs(lap1["lap_time_s"].iloc[0] - 90.5) < 1e-6

    def test_nat_lap_time_becomes_nan(self, bronze_laps_df: pd.DataFrame) -> None:
        silver = bronze_to_silver(bronze_laps_df, year=2024, round_number=1, session="R")
        # Lap 5 (pit lap) has NaT LapTime
        lap5 = silver[silver["lap_number"] == 5]
        assert np.isnan(lap5["lap_time_s"].iloc[0])

    def test_pit_flags(self, bronze_laps_df: pd.DataFrame) -> None:
        silver = bronze_to_silver(bronze_laps_df, year=2024, round_number=1, session="R")
        assert silver[silver["lap_number"] == 5]["pit_in_this_lap"].iloc[0] is True
        assert silver[silver["lap_number"] == 6]["pit_out_this_lap"].iloc[0] is True
        # All other laps should not be pit laps
        non_pit_in = silver[~silver["lap_number"].isin([5])]
        assert not non_pit_in["pit_in_this_lap"].any()

    def test_null_compound_becomes_unknown(self, bronze_laps_df: pd.DataFrame) -> None:
        bronze_laps_df = bronze_laps_df.copy()
        bronze_laps_df.loc[0, "Compound"] = None
        silver = bronze_to_silver(bronze_laps_df, year=2024, round_number=1, session="R")
        assert silver.loc[silver["lap_number"] == 1, "compound"].iloc[0] == "UNKNOWN"

    def test_pandera_validation_passes(self, bronze_laps_df: pd.DataFrame) -> None:
        from services.ingestion.schemas.silver import SilverLapSchema

        silver = bronze_to_silver(bronze_laps_df, year=2024, round_number=1, session="R")
        SilverLapSchema.validate(silver)  # should not raise


class TestSilverToGold:
    def test_race_progress_range(self, silver_laps_df: pd.DataFrame) -> None:
        gold = silver_to_gold(silver_laps_df, total_laps=57)
        assert gold["race_progress"].between(0.0, 1.0).all()

    def test_compound_encoded_values(self, silver_laps_df: pd.DataFrame) -> None:
        gold = silver_to_gold(silver_laps_df, total_laps=57)
        # Fixture has MEDIUM (0→1) and HARD (1→2)
        assert set(gold["compound_encoded"].unique()).issubset(set(COMPOUND_ENCODING.values()))

    def test_track_status_encoded_for_sc_laps(self, silver_laps_df: pd.DataFrame) -> None:
        gold = silver_to_gold(silver_laps_df, total_laps=57)
        # Laps 4-5 have TrackStatus='6' (VSC) → encoded as 3
        sc_laps = gold[gold["track_status"] == "6"]
        assert (sc_laps["track_status_encoded"] == 3).all()

    def test_laps_since_sc_increases_after_sc(self, silver_laps_df: pd.DataFrame) -> None:
        gold = silver_to_gold(silver_laps_df, total_laps=57)
        # After the VSC period (laps 4-5), laps_since_sc should increment
        post_sc = gold[gold["lap_number"] > 5]
        assert post_sc["sc_laps_since_last"].is_monotonic_increasing

    def test_tyre_deg_rate_non_negative(self, silver_laps_df: pd.DataFrame) -> None:
        gold = silver_to_gold(silver_laps_df, total_laps=57)
        valid = gold["tyre_deg_rate_s_per_lap"].dropna()
        assert (valid >= 0.0).all()

    def test_pandera_validation_passes(self, silver_laps_df: pd.DataFrame) -> None:
        from services.ingestion.schemas.gold import GoldLapSchema

        gold = silver_to_gold(silver_laps_df, total_laps=57)
        GoldLapSchema.validate(gold)  # should not raise


class TestHelpers:
    @pytest.mark.parametrize(
        ("status_str", "expected"),
        [
            ("1", 0),   # clear
            ("2", 1),   # yellow
            ("6", 3),   # SC
            ("5", 4),   # red flag
            ("12", 1),  # composite — highest priority wins
        ],
    )
    def test_encode_track_status(self, status_str: str, expected: int) -> None:
        series = pd.Series([status_str])
        result = _encode_track_status(series)
        assert result.iloc[0] == expected

    def test_encode_track_status_nan(self) -> None:
        series = pd.Series([None])
        result = _encode_track_status(series)
        assert result.iloc[0] == 0  # defaults to clear

    def test_laps_since_sc_before_any_sc(self) -> None:
        df = pd.DataFrame({"track_status": ["1", "1", "1"]})
        result = _laps_since_sc(df)
        assert result.isna().all()

    def test_laps_since_sc_resets_on_new_sc(self) -> None:
        df = pd.DataFrame({"track_status": ["1", "6", "1", "1", "6", "1"]})
        result = _laps_since_sc(df)
        # Lap index 2 is 1 lap after SC (index 1), lap index 4 resets to 0
        assert result.iloc[2] == 1.0
        assert result.iloc[4] == 0.0
        assert result.iloc[5] == 1.0
