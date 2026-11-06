"""Weekend calibration, session leakage and exact event identity contracts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import Mock

import pandas as pd
import pytest
from services.live import weekend
from services.live.weekend import WeekendConfig, load_weekend_summary, practice_calibration


def practice_rows():
    return pd.DataFrame(
        [
            {
                "session": session,
                "driver_number": str(driver),
                "stint_number": 1,
                "compound_encoded": 1,
                "is_accurate": True,
                "track_status_encoded": 0,
                "pit_in_this_lap": False,
                "pit_out_this_lap": False,
                "tyre_life_laps": age,
                "lap_time_s": intercept + 0.1 * age,
            }
            for session, driver, intercept in [("FP1", 1, 100), ("FP2", 1, 95), ("FP2", 2, 102)]
            for age in range(3, 12)
        ]
    )


def test_calibration_keeps_driver_session_intercepts_and_shrinks_to_prior():
    result = practice_calibration(practice_rows())
    evidence = result["compound_evidence"]["MEDIUM"]
    assert evidence["observed_slope_s_per_lap"] == pytest.approx(0.1)
    assert evidence["usable_stints"] == 3
    assert 0.0575 < result["degrade_rate"]["1"] < 0.1
    assert result["compound_evidence"]["HARD"]["source"] == "Austin circuit prior"


@pytest.mark.parametrize("session", ["Q", "R"])
def test_calibration_excludes_qualifying_and_race_outcomes(session):
    frame = practice_rows()
    poisoned = frame.copy()
    poisoned["session"] = session
    poisoned["lap_time_s"] = poisoned["tyre_life_laps"] * 0.49
    assert practice_calibration(pd.concat([frame, poisoned])) == practice_calibration(frame)


def test_weekend_summary_matches_exact_race_and_uses_shipped_snapshot(tmp_path, monkeypatch):
    config = {
        "year": 2026,
        "round_number": 18,
        "meeting_key": 1297,
        "race_session_key": 11396,
        "country_name": "United States",
        "circuit_short_name": "Austin",
        "circuit_reference_round": 19,
        "scheduled_laps": 56,
        "race_start_utc": "2026-10-25T20:00:00Z",
        "output_dir": str(tmp_path / "local"),
        "snapshot_path": str(tmp_path / "snapshot.json"),
        "sources": [],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    snapshot = {"meeting_key": 1297, "race_session": {"session_key": 11396}, "scheduled_laps": 56}
    (tmp_path / "snapshot.json").write_text(json.dumps(snapshot))
    monkeypatch.setattr(weekend, "DEFAULT_CONFIG", config_path)
    assert load_weekend_summary()["scheduled_laps"] == 56
    assert load_weekend_summary({"session_name": "Race", "meeting_key": 1297, "session_key": 11396})
    assert (
        load_weekend_summary({"session_name": "Race", "meeting_key": 1284, "session_key": 11280})
        is None
    )
    assert (
        load_weekend_summary(
            {"session_name": "Qualifying", "meeting_key": 1297, "session_key": 11392}
        )
        is None
    )


def test_refresh_rejects_changed_race_schedule_before_fetching_laps(tmp_path):
    config = WeekendConfig(
        year=2026,
        round_number=18,
        meeting_key=1297,
        race_session_key=11396,
        country_name="United States",
        circuit_short_name="Austin",
        circuit_reference_round=19,
        scheduled_laps=56,
        race_start_utc=datetime(2026, 10, 25, 20, tzinfo=UTC),
        output_dir=tmp_path,
        snapshot_path=tmp_path / "snapshot.json",
        sources=[],
    )
    feed = Mock()
    feed.get.return_value = [
        {
            "year": 2026,
            "session_name": "Race",
            "session_key": 11396,
            "circuit_short_name": "Austin",
            "date_start": "2026-10-25T21:00:00Z",
        }
    ]
    with pytest.raises(ValueError, match="Race start changed"):
        weekend.refresh_weekend(config, feed)
    assert feed.get.call_count == 1


@pytest.fixture
def acquisition_case(tmp_path, monkeypatch):
    config = WeekendConfig(
        year=2026,
        round_number=18,
        meeting_key=1297,
        race_session_key=11396,
        country_name="United States",
        circuit_short_name="Austin",
        circuit_reference_round=19,
        scheduled_laps=56,
        race_start_utc=datetime(2026, 10, 25, 20, tzinfo=UTC),
        output_dir=tmp_path / "weekend",
        snapshot_path=tmp_path / "snapshot.json",
        sources=["https://api.openf1.org/v1/sessions?meeting_key=1297"],
    )
    sessions = [
        {
            "year": 2026,
            "meeting_key": 1297,
            "session_key": key,
            "session_name": name,
            "circuit_short_name": "Austin",
            "date_start": begin,
            "date_end": end,
        }
        for key, name, begin, end in [
            (11388, "Practice 1", "2026-10-23T17:00:00Z", "2026-10-23T18:00:00Z"),
            (11390, "Practice 2", "2026-10-23T21:00:00Z", "2026-10-23T22:00:00Z"),
            (11391, "Practice 3", "2026-10-24T17:00:00Z", "2026-10-24T18:00:00Z"),
            (11392, "Qualifying", "2026-10-24T20:00:00Z", "2026-10-24T21:00:00Z"),
            (11396, "Race", "2026-10-25T20:00:00Z", "2026-10-25T22:00:00Z"),
        ]
    ]
    roster = [
        {"driver_number": 1, "name_acronym": "AAA", "team_name": "Team A"},
        {"driver_number": 2, "name_acronym": "BBB", "team_name": "Team B"},
    ]
    qualifying = [{"position": 1, "driver_number": 1}, {"position": 2, "driver_number": 2}]
    grid = [{"position": 2, "driver_number": 1}, {"position": 1, "driver_number": 2}]

    def get(endpoint, params, **kwargs):
        if endpoint == "sessions":
            return sessions
        if endpoint == "drivers":
            return roster
        if endpoint == "session_result":
            return qualifying
        if endpoint == "starting_grid":
            return grid
        return []

    feed = Mock()
    feed.get.side_effect = get
    # Isolate acquisition selection/publication from the separately tested transforms.
    monkeypatch.setattr(weekend, "canonical_laps", Mock(return_value=pd.DataFrame()))
    monkeypatch.setattr(weekend.BronzeLapSchema, "validate", Mock())
    monkeypatch.setattr(weekend, "bronze_to_silver", Mock())
    monkeypatch.setattr(weekend, "silver_to_gold", Mock(return_value=practice_rows()))
    return config, feed


def test_refresh_never_fetches_unfinished_qualifying_or_future_race(acquisition_case):
    config, feed = acquisition_case
    result = weekend.refresh_weekend(config, feed, as_of=datetime(2026, 10, 24, 20, 30, tzinfo=UTC))
    assert [entry["code"] for entry in result["sessions"]] == ["FP1", "FP2", "FP3"]
    session_requests = [
        call.args[1]["session_key"]
        for call in feed.get.call_args_list
        if "session_key" in call.args[1]
    ]
    assert 11392 not in session_requests
    assert 11396 not in session_requests
    assert result["qualifying"] == []
    assert result["status"] == "partial"


def test_refresh_retains_qualifying_separately_from_meeting_starting_grid(acquisition_case):
    config, feed = acquisition_case
    result = weekend.refresh_weekend(config, feed, as_of=datetime(2026, 10, 24, 22, tzinfo=UTC))
    assert result["status"] == "complete"
    assert [row["driver_number"] for row in result["qualifying"]] == [1, 2]
    assert [row["driver_number"] for row in result["starting_grid"]] == [2, 1]
    assert result["starting_grid"][0]["name_acronym"] == "BBB"
    assert result["grid_status"] == "published starting grid"
    grid_request = next(call for call in feed.get.call_args_list if call.args[0] == "starting_grid")
    assert grid_request.args[1] == {"meeting_key": 1297}
    assert grid_request.kwargs["refresh"] is True
    for session in result["sessions"]:
        for artifact in session["files"].values():
            assert len(artifact["sha256"]) == 64
    published = json.loads(config.snapshot_path.read_text())
    assert published["qualifying"] == result["qualifying"]
    assert published["starting_grid"] == result["starting_grid"]


def test_pending_grid_is_not_presented_as_published_qualifying_order(acquisition_case):
    config, feed = acquisition_case
    fallback = feed.get.side_effect

    def get(endpoint, params, **kwargs):
        return [] if endpoint == "starting_grid" else fallback(endpoint, params, **kwargs)

    feed.get.side_effect = get
    result = weekend.refresh_weekend(config, feed, as_of=datetime(2026, 10, 24, 22, tzinfo=UTC))
    assert len(result["qualifying"]) == 2
    assert result["starting_grid"] == []
    assert result["grid_status"] == "qualifying order; grid pending"
