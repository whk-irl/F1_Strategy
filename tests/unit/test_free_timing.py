"""Contracts for the experimental public SignalR timing adapter (no network)."""

from __future__ import annotations

import io
import json
import subprocess
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
import requests
from services.live import free_timing as timing
from signalrcore.messages.completion_message import CompletionMessage

NOW = datetime(2026, 10, 25, 19, 5, tzinfo=UTC)


def session():
    return {
        "Key": 11201,
        "Name": "Race",
        "StartDate": "2026-10-25T14:00:00",
        "EndDate": "2026-10-25T16:00:00",
        "GmtOffset": "-05:00:00",
        "Meeting": {"Name": "United States Grand Prix", "Circuit": {"ShortName": "Austin"}},
    }


def state():
    result = timing.FreeTimingState()
    result.connected = True
    result.update("SessionInfo", session(), now=NOW)
    result.update("DriverList", {"1": {"Tla": "VER", "TeamName": "Red Bull"}}, now=NOW)
    result.update(
        "TimingData",
        {
            "Lines": {
                "1": {
                    "NumberOfLaps": 3,
                    "Position": "2",
                    "LastLapTime": {"Value": "1:39.125"},
                    "GapToLeader": "+2.5",
                    "IntervalToPositionAhead": {"Value": "+2.5"},
                    "InPit": False,
                    "NumberOfPitStops": 0,
                }
            }
        },
        NOW.isoformat(),
        now=NOW,
    )
    return result


def test_partial_updates_preserve_driver_fields_and_absent_values():
    result = state()
    result.update("TimingData", {"Lines": {"1": {"Position": "1"}}}, NOW.isoformat(), now=NOW)
    driver = result.snapshot()["drivers"][0]
    assert driver["position"] == 1
    assert driver["completed_laps"] == 3
    assert driver["last_lap_duration"] == 99.125
    assert driver["team_name"] == "Red Bull"
    assert driver["compound"] is None
    assert driver["tyre_age"] is None
    assert driver["in_pit"] is False


def test_initial_snapshots_and_heartbeats_do_not_establish_timing_freshness():
    result = timing.FreeTimingState()
    result.update("TimingData", {"Lines": {"1": {"Position": "1"}}}, now=NOW)
    result.update("Heartbeat", {"Utc": NOW.isoformat()}, NOW.isoformat(), now=NOW)
    snapshot = result.snapshot()
    assert snapshot["observed_at"] is None
    assert snapshot["stream_at"] == NOW.isoformat()
    result.update("TimingData", {"Lines": {}}, NOW.isoformat(), now=NOW)
    assert result.snapshot()["observed_at"] is None


def test_out_of_order_delta_does_not_overwrite_current_sample():
    result = state()
    accepted = result.update(
        "TimingData",
        {"Lines": {"1": {"Position": "9"}}},
        (NOW - timedelta(seconds=1)).isoformat(),
        now=NOW,
    )
    assert not accepted
    assert result.snapshot()["drivers"][0]["position"] == 2


def test_other_drivers_and_position_updates_do_not_refresh_old_gaps():
    result = state()
    later = NOW + timedelta(seconds=50)
    result.update("TimingData", {"Lines": {"44": {"Position": "3"}}}, later.isoformat(), now=later)
    first = result.snapshot()["drivers"][0]
    assert first["observed_at"] == NOW.isoformat()
    assert first["gap_to_leader_updated_at"] == NOW.isoformat()
    result.update("TimingData", {"Lines": {"1": {"Position": "1"}}}, later.isoformat(), now=later)
    first = result.snapshot()["drivers"][0]
    assert first["observed_at"] == later.isoformat()
    assert first["gap_to_leader_updated_at"] == NOW.isoformat()
    assert first["interval_updated_at"] == NOW.isoformat()


def test_stint_list_and_indexed_deltas_retain_prior_compounds():
    result = state()
    result.update(
        "TimingAppData",
        {
            "Lines": {
                "1": {
                    "Stints": [
                        {"Compound": "MEDIUM", "TotalLaps": 3, "New": "true"},
                    ]
                }
            }
        },
        now=NOW,
    )
    result.update(
        "TimingAppData",
        {
            "Lines": {
                "1": {
                    "Stints": {
                        "1": {"Compound": "HARD", "New": "true", "TotalLaps": 0},
                    }
                }
            }
        },
        NOW.isoformat(),
        now=NOW,
    )
    result.update(
        "TimingAppData",
        {
            "Lines": {
                "1": {
                    "Stints": {
                        "1": {"TotalLaps": 2},
                    }
                }
            }
        },
        NOW.isoformat(),
        now=NOW,
    )
    snapshot = result.snapshot()
    assert [stint["compound"] for stint in snapshot["stints"]] == ["MEDIUM", "HARD"]
    assert [stint["lap_start"] for stint in snapshot["stints"]] == [1, 4]
    assert snapshot["drivers"][0]["tyre_age"] == 2
    assert snapshot["drivers"][0]["compound"] == "HARD"


def test_late_join_does_not_invent_historical_stint_starts():
    result = state()
    result.update(
        "TimingAppData",
        {
            "Lines": {
                "1": {
                    "Stints": {
                        "1": {"Compound": "HARD", "LapNumber": 25, "TotalLaps": 7},
                    }
                }
            }
        },
        now=NOW,
    )
    stint = result.snapshot()["stints"][0]
    assert stint["lap_start"] is None
    assert stint["tyre_age_at_start"] is None


def test_lap_count_weather_control_and_no_unavailable_telemetry():
    result = state()
    result.update("LapCount", {"CurrentLap": 4, "TotalLaps": 56}, NOW.isoformat(), now=NOW)
    result.update("WeatherData", {"AirTemp": "29.5", "Rainfall": "0"}, NOW.isoformat(), now=NOW)
    result.update(
        "RaceControlMessages",
        {
            "Messages": [
                {
                    "Utc": NOW.isoformat(),
                    "Message": "VIRTUAL SAFETY CAR DEPLOYED",
                    "Flag": "VSC",
                }
            ]
        },
        NOW.isoformat(),
        now=NOW,
    )
    assert not result.update("Position.z", "not-decoded", NOW.isoformat(), now=NOW)
    assert not result.update("CarData.z", "not-decoded", NOW.isoformat(), now=NOW)
    snapshot = result.snapshot()
    assert snapshot["total_laps"] == 56
    assert snapshot["weather"]["air_temperature"] == 29.5
    assert snapshot["weather"]["track_temperature"] is None
    assert snapshot["race_control"][0]["date"] == NOW.isoformat()
    assert not snapshot["capabilities"]["car_data"]
    assert not snapshot["capabilities"]["position"]


def test_new_session_resets_old_driver_and_tyre_state():
    result = state()
    result.update("SessionInfo", {**session(), "Key": 11202}, now=NOW)
    snapshot = result.snapshot()
    assert snapshot["drivers"] == []
    assert snapshot["observed_at"] is None
    assert snapshot["connected"]


def test_session_dates_apply_separate_gmt_offset_only_to_naive_dates():
    assert timing._session_info(session())["date_start"] == "2026-10-25T19:00:00+00:00"
    explicit = {**session(), "StartDate": "2026-10-25T19:00:00Z"}
    assert timing._session_info(explicit)["date_start"] == "2026-10-25T19:00:00+00:00"


def test_atomic_snapshot_validates_freshness_and_session_identity(tmp_path):
    path = tmp_path / "snapshot.json"
    timing.write_snapshot(path, state().snapshot())
    expected = {
        "year": 2026,
        "session_name": "Race",
        "circuit_short_name": "Austin",
        "date_start": "2026-10-25T19:00:00Z",
        "session_key": 9999,
    }
    current = timing.load_free_snapshot(path, expected_meta=expected, now=NOW)
    assert current["sample_age_s"] == 0
    assert timing.load_free_snapshot(path, now=NOW + timedelta(seconds=46)) is None
    assert timing.load_free_snapshot(path, now=NOW + timedelta(seconds=46), max_age_s=None)
    for mismatch in (
        {"year": 2025},
        {"session_name": "Qualifying"},
        {"circuit_short_name": "Mexico City"},
        {"date_start": "2026-10-26T19:00:00Z"},
    ):
        assert (
            timing.load_free_snapshot(path, expected_meta={**expected, **mismatch}, now=NOW) is None
        )
    assert not path.with_suffix(".tmp.json").exists()


@pytest.mark.parametrize(
    "content", ["bad json", "[]", '{"schema_version": 2}', '{"schema_version":1,"source":"other"}']
)
def test_corrupt_or_unrecognized_snapshot_is_not_consumed(tmp_path, content):
    path = tmp_path / "snapshot.json"
    path.write_text(content)
    assert timing.load_free_snapshot(path, max_age_s=None, now=NOW) is None
    assert timing.load_free_snapshot(tmp_path / "missing.json") is None


def test_client_requests_explicit_no_auth_without_gps_or_car_data(tmp_path, monkeypatch):
    class FakeUpstream:
        def __init__(self, filename, **kwargs):
            self.options = kwargs
            self._output_file = io.StringIO()

        def _on_message(self, msg):
            self._output_file.write(str(msg))

    monkeypatch.setattr(timing, "_upstream_client", lambda: FakeUpstream)
    raw, snapshot = tmp_path / "raw.txt", tmp_path / "snapshot.json"
    client = timing.FreeTimingClient(raw, snapshot)
    assert client.client.options["no_auth"] is True
    assert client.client.options["timeout"] == 0
    assert "Position.z" not in client.client.topics
    assert "CarData.z" not in client.client.topics
    client.client._on_message(["TimingData", {"Lines": {"1": {"Position": "2"}}}, NOW.isoformat()])
    assert "TimingData" in client.client._output_file.getvalue()
    assert json.loads(snapshot.read_text())["drivers"][0]["position"] == 2
    completion = MagicMock(spec=CompletionMessage)
    completion.result = {"DriverList": {"1": {"Tla": "VER"}}}
    client.client._on_message(completion)
    assert json.loads(snapshot.read_text())["drivers"][0]["name_acronym"] == "VER"


def test_public_connection_omits_optional_auth_factory_and_accepts_options405(
    tmp_path, monkeypatch
):
    class FakeUpstream:
        _connection_url = "wss://example.invalid/timing"
        _negotiate_url = "https://example.invalid/negotiate"

        def __init__(self, filename, filemode, **kwargs):
            self.filename, self.filemode = filename, filemode
            self.headers, self._is_connected = {}, False

        def _on_connect(self):
            self._is_connected = True

        def _on_close(self):
            self._is_connected = False

    monkeypatch.setattr(timing, "_upstream_client", lambda: FakeUpstream)
    response = MagicMock(status_code=405, cookies={"AWSALBCORS": "routing-only-cookie"})
    options = MagicMock(return_value=response)
    monkeypatch.setattr(timing.requests, "options", options)
    connection, builder = MagicMock(), MagicMock()
    builder.with_url.return_value = builder
    builder.build.return_value = connection
    monkeypatch.setattr(timing, "HubConnectionBuilder", lambda: builder)
    collector = timing.FreeTimingClient(tmp_path / "raw.txt", tmp_path / "snapshot.json")
    connection.start.side_effect = collector.client._on_connect
    collector.client._run()
    connection_options = builder.with_url.call_args.kwargs["options"]
    assert "access_token_factory" not in connection_options
    assert "Authorization" not in connection_options["headers"]
    assert connection_options["verify_ssl"] is True
    assert options.call_args.kwargs["timeout"] == 10
    assert not response.raise_for_status.called
    assert collector.connected_once
    collector._files.close()


def test_auth_denial_stops_without_retries(tmp_path, monkeypatch):
    class FakeUpstream:
        def __init__(self, *args, **kwargs):
            pass

        def _run(self):
            response = MagicMock(status_code=403)
            raise requests.HTTPError("private provider response", response=response)

    monkeypatch.setattr(timing, "_upstream_client", lambda: FakeUpstream)
    sleep = MagicMock()
    monkeypatch.setattr(timing.time, "sleep", sleep)
    collector = timing.FreeTimingClient(tmp_path / "raw.txt", tmp_path / "snapshot.json")
    monkeypatch.setattr(
        collector.client,
        "_run",
        MagicMock(side_effect=requests.HTTPError("private", response=MagicMock(status_code=403))),
    )
    with pytest.raises(PermissionError, match="Public provider access denied"):
        collector.start()
    sleep.assert_not_called()


def test_keyboard_interrupt_stops_without_reconnecting(tmp_path, monkeypatch):
    class FakeUpstream:
        def __init__(self, *args, **kwargs):
            pass

        def _run(self):
            raise KeyboardInterrupt

    monkeypatch.setattr(timing, "_upstream_client", lambda: FakeUpstream)
    collector = timing.FreeTimingClient(tmp_path / "raw.txt", tmp_path / "snapshot.json")
    monkeypatch.setattr(collector.client, "_run", MagicMock(side_effect=KeyboardInterrupt))
    sleep = MagicMock()
    monkeypatch.setattr(timing.time, "sleep", sleep)
    collector.start()
    sleep.assert_not_called()


def test_probe_is_bounded_and_does_not_expose_exception_details(monkeypatch):
    runner = MagicMock(side_effect=subprocess.TimeoutExpired("sensitive-url", timeout=11))
    monkeypatch.setattr(timing.subprocess, "run", runner)
    report = timing.probe_free_timing(1)
    assert report["status"] == "failed"
    assert "sensitive" not in json.dumps(report)
    assert runner.call_args.kwargs["timeout"] == 11
    with pytest.raises(ValueError, match="between 1 and 60"):
        timing.probe_free_timing(61)


def test_probe_uses_safe_child_result_and_waits_for_timing(monkeypatch):
    report = timing._probe_report(state().snapshot(), connected=True)
    assert report["status"] == "waiting"  # raw state has no computed freshness age
    snapshot = state().snapshot()
    snapshot["sample_age_s"] = 1
    report = timing._probe_report(snapshot, connected=True)
    assert report["status"] == "ready"
    runner = MagicMock(return_value=MagicMock(stdout="log line\n" + json.dumps(report)))
    monkeypatch.setattr(timing.subprocess, "run", runner)
    assert timing.probe_free_timing(3) == report


def test_parser_rejects_malformed_payload_and_nonfinite_values():
    result = timing.FreeTimingState()
    assert not result.update("TimingData", "not json")
    assert not result.update("TimingData", [])
    assert result.update("TimingData", '{"Lines":{"1":{"Position":"1"}}}')
    assert timing._number("inf") is None
    assert timing._seconds("1:inf") is None
    assert timing._seconds("1:bad") is None
