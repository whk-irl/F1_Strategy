"""Unit tests for services/live/timing_state.py."""

from __future__ import annotations

import pytest
from services.live.timing_state import (
    TimingSnapshot,
    deep_merge,
    parse_lap_time_s,
)


class TestParseLapTime:
    def test_minutes_seconds(self) -> None:
        assert parse_lap_time_s("1:23.456") == pytest.approx(83.456)

    def test_seconds_only(self) -> None:
        assert parse_lap_time_s("92.310") == pytest.approx(92.31)

    def test_numeric(self) -> None:
        assert parse_lap_time_s(90.5) == pytest.approx(90.5)

    def test_none(self) -> None:
        assert parse_lap_time_s(None) is None


class TestDeepMerge:
    def test_nested_delta(self) -> None:
        base = {"Lines": {"44": {"Position": "1", "NumberOfLaps": "10"}}}
        update = {"Lines": {"44": {"NumberOfLaps": "11", "LastLapTime": {"Value": "1:30.0"}}}}
        deep_merge(base, update)
        assert base["Lines"]["44"]["Position"] == "1"
        assert base["Lines"]["44"]["NumberOfLaps"] == "11"
        assert base["Lines"]["44"]["LastLapTime"]["Value"] == "1:30.0"


class TestTimingSnapshot:
    def _sample_snapshot(self) -> TimingSnapshot:
        snap = TimingSnapshot()
        snap.apply_topic(
            "SessionInfo",
            {
                "Meeting": {
                    "Key": 1219,
                    "Name": "Monaco Grand Prix",
                    "Country": {"Name": "Monaco"},
                    "Circuit": {"ShortName": "Monaco"},
                },
                "Key": 9999,
                "Name": "Race",
                "SessionStatus": "Started",
                "StartDate": "2026-05-25T13:00:00Z",
            },
        )
        snap.apply_topic("LapCount", {"CurrentLap": "12", "TotalLaps": "78"})
        snap.apply_topic(
            "DriverList",
            {
                "44": {"RacingNumber": "44", "Tla": "HAM", "TeamName": "Ferrari"},
                "1": {"RacingNumber": "1", "Tla": "VER", "TeamName": "Red Bull"},
            },
        )
        snap.apply_topic(
            "TimingData",
            {
                "Lines": {
                    "44": {
                        "Position": "2",
                        "NumberOfLaps": "12",
                        "NumberOfPitStops": "1",
                        "LastLapTime": {"Value": "1:15.123"},
                    }
                }
            },
        )
        snap.apply_topic(
            "TimingAppData",
            {
                "Lines": {
                    "44": {
                        "Stints": {
                            "0": {"Compound": "MEDIUM", "Start": "1", "TotalLaps": "12"},
                        }
                    }
                }
            },
        )
        snap.apply_topic("WeatherData", {"TrackTemp": "35.5", "Rainfall": "0"})
        snap.has_timing_data = True
        return snap

    def test_session_meta(self) -> None:
        meta = self._sample_snapshot().session_meta()
        assert meta is not None
        assert meta["session_key"] == 9999
        assert meta["meeting_key"] == 1219
        assert meta["meeting_name"] == "Monaco Grand Prix"
        assert meta["total_laps"] == 78
        assert meta["session_name"] == "Race"

    def test_drivers(self) -> None:
        drivers = self._sample_snapshot().drivers()
        assert len(drivers) == 2
        assert drivers[0]["driver_number"] == 1

    def test_latest_lap(self) -> None:
        lap = self._sample_snapshot().latest_lap(44)
        assert lap is not None
        assert lap["lap_number"] == 12
        assert lap["lap_duration"] == pytest.approx(75.123)

    def test_position_and_pits(self) -> None:
        snap = self._sample_snapshot()
        assert snap.latest_position(44) == 2
        assert snap.pit_stop_count(44) == 1

    def test_current_stint(self) -> None:
        stint = self._sample_snapshot().current_stint(44, 12)
        assert stint is not None
        assert stint["compound"] == "MEDIUM"
        assert stint["lap_start"] == 1

    def test_weather_mapped(self) -> None:
        weather = self._sample_snapshot().latest_weather()
        assert weather is not None
        assert weather["track_temperature"] == pytest.approx(35.5)

    def test_sc_from_track_status(self) -> None:
        snap = TimingSnapshot()
        snap.apply_topic("TrackStatus", {"Status": "4", "Message": "SC DEPLOYED"})
        assert snap.is_safety_car_active() is True

    def test_all_stints(self) -> None:
        stints = self._sample_snapshot().all_stints()
        assert len(stints) == 1
        assert stints[0]["driver_number"] == 44

    def test_session_finalised(self) -> None:
        snap = TimingSnapshot()
        snap.apply_topic(
            "SessionInfo",
            {
                "Meeting": {"Key": 1, "Name": "Test GP"},
                "Name": "Qualifying",
                "SessionStatus": "Finalised",
                "ArchiveStatus": {"Status": "Complete"},
            },
        )
        assert snap.is_session_finalised() is True

    def test_stints_list_format_from_free_feed(self) -> None:
        """Free formula1.com feed sends Stints as a list, not a numbered dict."""
        snap = TimingSnapshot()
        snap.apply_topic(
            "TimingAppData",
            {
                "Lines": {
                    "44": {
                        "Stints": [
                            {
                                "Compound": "MEDIUM",
                                "StartLaps": 0,
                                "TotalLaps": 23,
                            }
                        ]
                    }
                }
            },
        )
        stints = snap.stints_for_driver(44)
        assert len(stints) == 1
        assert stints[0]["compound"] == "MEDIUM"
        assert stints[0]["lap_start"] == 1
        assert stints[0]["lap_end"] == 23
