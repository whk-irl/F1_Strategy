"""Regression checks for temporal leakage, lifecycle boundaries and pit semantics."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from ml.models.safety_car.train import SC_HORIZON, TARGET_COL, _build_sc_label
from ml.models.tire_degradation.sequence_dataset import FEATURE_COLS, _build_windows
from ml.pipelines.config import TrainingConfig
from ml.pipelines.data import split_events
from ml.pipelines.policy import DynamicProgramming, HistoryEnv, historical_actions
from services.ingestion.acquire import completed_events, join_weather
from services.ingestion.transforms import bronze_to_silver, silver_to_gold
from services.simulator.env import F1RaceEnv


class ConstantModel:
    def __init__(self, value=0.0):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value)


@pytest.fixture
def gold(silver_laps_df):
    return silver_to_gold(silver_laps_df, 10)


def test_features_are_prefix_invariant(silver_laps_df):
    changed = silver_laps_df.copy()
    changed.loc[changed["lap_number"] > 7, "lap_time_s"] = 300.0
    a, b = silver_to_gold(silver_laps_df, 10), silver_to_gold(changed, 10)
    columns = ["lap_time_delta_s", "tyre_deg_rate_s_per_lap", "rolling_lap_time_3_s"]
    pd.testing.assert_frame_equal(
        a.loc[a["lap_number"] <= 7, columns], b.loc[b["lap_number"] <= 7, columns]
    )


def test_shuffled_laps_have_same_features(silver_laps_df):
    expected = silver_to_gold(silver_laps_df, 10).sort_index()
    actual = silver_to_gold(silver_laps_df.sample(frac=1, random_state=9), 10).sort_index()
    pd.testing.assert_frame_equal(actual, expected)


def test_sc_counter_counts_laps_once_per_field(silver_laps_df):
    second = silver_laps_df.copy()
    second["driver_number"], second["driver_code"] = "44", "HAM"
    frame = pd.concat([silver_laps_df, second], ignore_index=True)
    result = silver_to_gold(frame, 10)
    assert result.loc[result["lap_number"] == 6, "sc_laps_since_last"].eq(1).all()


def test_actual_completion_timestamps(silver_laps_df):
    frame = silver_laps_df.copy()
    frame["lap_end_time_s"] = frame["lap_number"] * 90.0
    origin = pd.Timestamp("2026-03-08T04:00:00Z")
    result = silver_to_gold(frame, 10, session_start=origin)
    assert result.iloc[0]["event_timestamp"] == origin + pd.Timedelta(seconds=90)


def test_missing_optional_pit_columns_are_false(bronze_laps_df):
    result = bronze_to_silver(
        bronze_laps_df.drop(columns=["PitInTime", "PitOutTime"]), 2024, 1, "R"
    )
    assert not result["pit_in_this_lap"].any()
    assert not result["pit_out_this_lap"].any()


def test_weather_join_does_not_use_future_sample():
    laps = pd.DataFrame({"LapStartTime": pd.to_timedelta([10, 20], unit="s")})
    weather = pd.DataFrame(
        {"Time": pd.to_timedelta([5, 15, 25], unit="s"), "TrackTemp": [30, 31, 99]}
    )
    assert join_weather(laps, weather)["track_temp_c"].tolist() == [30, 31]


def test_calendar_excludes_future_and_running_events():
    schedule = pd.DataFrame(
        {"RoundNumber": [1, 2, 3], "Session5DateUtc": ["2026-10-01", "2026-10-15", "2026-10-20"]}
    )
    selected = completed_events(schedule, datetime(2026, 10, 15, 18, tzinfo=UTC))
    assert selected["RoundNumber"].tolist() == [1]


def test_sc_horizon_is_strictly_future():
    frame = pd.DataFrame(
        {
            "year": 2024,
            "round_number": 1,
            "lap_number": range(1, 7),
            "track_status_encoded": [0, 0, 0, 2, 0, 0],
        }
    )
    labels = _build_sc_label(frame, SC_HORIZON)[TARGET_COL].tolist()
    assert labels[:4] == [1, 1, 1, 0]


def test_red_flag_is_not_sc_onset():
    frame = pd.DataFrame(
        {"year": 2024, "round_number": 1, "lap_number": [1, 2], "track_status_encoded": [0, 4]}
    )
    assert not _build_sc_label(frame)[TARGET_COL].any()


def test_sequence_windows_break_at_missing_lap():
    frame = pd.DataFrame(np.ones((6, len(FEATURE_COLS))), columns=FEATURE_COLS)
    frame["lap_number"] = [1, 2, 3, 7, 8, 9]
    windows, _ = _build_windows(frame, FEATURE_COLS, "lap_time_delta_s", 2)
    assert len(windows) == 2


def test_split_has_chronological_recent_holdouts():
    rows = [(2022, 1), (2022, 5), (2024, 1), (2025, 24), (2026, 10), (2026, 11), (2026, 14)]
    frame = pd.DataFrame(rows, columns=["year", "round_number"]).assign(session="R")
    splits = split_events(frame, TrainingConfig())
    assert splits["finetune"]["round_number"].tolist() == [24, 10]
    assert splits["finetune_val"]["round_number"].tolist() == [11]
    assert splits["test"]["round_number"].tolist() == [14]


def test_overlap_config_is_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        TrainingConfig(finetune_years=[2023])


def test_empty_split_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        split_events(
            pd.DataFrame({"year": [2022], "round_number": [1], "session": ["R"]}), TrainingConfig()
        )


def test_pit_label_uses_next_compound(gold):
    assert historical_actions(gold)[5] == 3


def test_pit_tyre_change_applies_to_next_lap(gold):
    env = F1RaceEnv(gold, "1", ConstantModel(), noise_std=0.0)
    env.reset(seed=3)
    _, _, _, _, info = env.step(3)
    assert info["tyre_life"] == 1
    assert env._tyre_life == 1
    assert env._compound == 2


def test_replay_does_not_double_charge_pit_loss(gold):
    env = F1RaceEnv(gold, "1", ConstantModel(), replay_mode=True)
    env.reset(seed=3)
    _, _, _, _, info = env.step(3)
    assert info["sim_lap_time_s"] == pytest.approx(90.5)


def test_tire_cache_includes_model_identity(gold):
    first = F1RaceEnv(gold, "1", ConstantModel(0.0))
    second = F1RaceEnv(gold, "1", ConstantModel(2.0))
    assert not np.array_equal(first._tire_table, second._tire_table)


def test_history_does_not_cross_episode_boundary(gold):
    env = HistoryEnv(F1RaceEnv(gold, "1", ConstantModel(), noise_std=0.0), frames=4)
    first, _ = env.reset(seed=3)
    assert first.shape == (96,)
    assert not first[:72].any()
    env.step(1)
    reset, _ = env.reset(seed=3)
    np.testing.assert_array_equal(first, reset)


def test_dynamic_programming_requires_two_compounds(gold):
    env = F1RaceEnv(gold, "1", ConstantModel(), noise_std=0.0)
    env.reset(seed=3)
    planner = DynamicProgramming(env)
    for _ in range(10):
        env.step(planner.action())
    assert len(env._used_compounds) >= 2
    planner.value.cache_clear()
