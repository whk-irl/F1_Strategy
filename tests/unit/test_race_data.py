"""Race-feed contracts: timestamps, bounds, missing values and recoverable archives."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests
from services.live import race_data as data


@pytest.fixture
def race_records():
    start = pd.Timestamp("2024-09-01T13:03:00Z")
    meta = {
        "session_key": 1,
        "date_start": "2024-09-01T13:00:00Z",
        "date_end": "2024-09-01T15:00:00Z",
    }
    laps = [
        {
            "driver_number": 1,
            "lap_number": i + 1,
            "date_start": (start + pd.Timedelta(seconds=i * 90)).isoformat(),
            "lap_duration": 90,
            "duration_sector_1": 29,
            "duration_sector_2": 30,
            "duration_sector_3": 31,
        }
        for i in range(2)
    ]
    locations = [
        {
            "driver_number": 1,
            "date": (start + pd.Timedelta(seconds=i)).isoformat(),
            "x": i + 1,
            "y": i + 10,
        }
        for i in range(-10, 200)
    ]
    records = {
        "laps": laps,
        "drivers": [{"driver_number": 1}],
        "position": [],
        "intervals": [],
        "stints": [],
        "pit": [],
        "weather": [],
        "race_control": [],
    }
    return meta, records, locations, start


def test_filter_url_uses_provider_comparison_syntax():
    url = data.filtered_url(
        "location", {"session_key": 1}, "2024-09-01T13:00:00+00:00", "2024-09-01T13:01:00Z"
    )
    assert "date%3E2024" in url
    assert "date%3E=" not in url
    assert "%2B00:00" in url
    assert "date%3C2024" in url


def test_series_bounds_sort_and_missing_samples(race_records):
    _, _, _, start = race_records
    rows = [
        {"driver_number": 1, "date": (start + pd.Timedelta(seconds=t)).isoformat(), "speed": value}
        for t, value in [(2, 200), (-1, 300), (1, None), (3, 250), (1, None)]
    ]
    result = data.series_by_driver(rows, start, ["speed"], end=start + pd.Timedelta(seconds=2))
    assert result == {"1": [[1.0, None], [2.0, 200.0]]}
    assert data.series_by_driver([], start, ["speed"]) == {}
    assert data.numeric(float("inf")) is None
    assert data.numeric("+1 LAP") is None
    assert data.numeric("3.5") == 3.5
    assert data.timestamp("2024-09-01").tzinfo is not None


def test_bundle_uses_actual_start_and_timed_sector_locations(race_records):
    meta, records, locations, start = race_records
    locations.append({"driver_number": 241, "date": start.isoformat(), "x": 2, "y": 5})
    records["weather"] = [
        {"date": (start - pd.Timedelta(seconds=1)).isoformat(), "rainfall": 1},
        {"date": start.isoformat(), "rainfall": 0},
    ]
    bundle = data.build_bundle(meta, records, locations)
    assert data.timestamp(bundle["origin"]) == start
    assert bundle["duration"] == 180
    assert bundle["total_laps"] == 2
    assert set(bundle["locations"]) == {"1"}
    assert all(0 <= row[0] <= 180 for row in bundle["locations"]["1"])
    assert len(bundle["track"]) >= 50
    assert bundle["sectors"][0] == {"label": "S1", "point": [30, 39]}
    assert bundle["sectors"][1] == {"label": "S2", "point": [60, 69]}
    assert len(bundle["weather"]) == 1
    assert bundle["laps"]["1"][0][1] == 90
    json.dumps(bundle, allow_nan=False)


def test_bundle_requires_start_and_complete_trace(race_records):
    meta, records, _, _ = race_records
    with pytest.raises(data.RaceFeedError, match="complete recorded lap"):
        data.build_bundle(meta, records, [])
    assert data.build_bundle(meta, records, [], track_fallback=[[1, 2]])["track"] == [[1, 2]]
    records["laps"] = []
    with pytest.raises(data.RaceFeedError, match="race start"):
        data.build_bundle(meta, records, [])


def test_archive_preserves_mixed_gap_types_and_avoids_second_request(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CACHE", tmp_path)
    monkeypatch.setattr(data, "ARCHIVE", tmp_path / "legacy")
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    feed = data.RaceFeed(token="test-token")
    rows = [{"interval": 1.5}, {"interval": "+1 LAP"}, {"interval": None}]
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = rows
    feed.http.get = MagicMock(return_value=response)
    assert feed.get("intervals", {"session_key": 1}, archive=True, ttl=60) == rows
    assert feed.get("intervals", {"session_key": 1}, archive=True) == rows
    assert feed.http.get.call_count == 1
    assert "test-token" not in next((tmp_path / "responses").glob("*.json")).read_text()
    assert feed.get("intervals", {"session_key": 1}, ttl=60) == rows
    assert feed.http.get.call_count == 1


def test_legacy_archive_and_finished_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ARCHIVE", tmp_path)
    monkeypatch.setattr(data, "CACHE", tmp_path / "other")
    rows = [
        {
            "session_key": 1,
            "date_start": "2024-01-01T12:00:00Z",
            "date_end": "2024-01-01T15:00:00Z",
        },
        {
            "session_key": 2,
            "date_start": "2024-02-01T12:00:00Z",
            "date_end": "2024-02-01T15:00:00Z",
            "is_cancelled": True,
        },
        {
            "session_key": 3,
            "date_start": "2099-02-01T12:00:00Z",
            "date_end": "2099-02-01T15:00:00Z",
        },
    ]
    (tmp_path / "sessions_session_name-Race_year-2024.json").write_text(json.dumps(rows))
    feed = data.RaceFeed()
    feed.http.get = MagicMock(side_effect=requests.ConnectionError("offline"))
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    assert [s["session_key"] for s in data.session_catalog(feed, 2024)] == [1]


def test_catalog_refreshes_calendar_and_bulk_archive_does_not_grow_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CACHE", tmp_path)
    monkeypatch.setattr(data, "ARCHIVE", tmp_path / "legacy")
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    feed = data.RaceFeed()
    response = MagicMock(status_code=200)
    response.json.return_value = [
        {"session_key": 1, "date_start": "2024-01-01T12:00:00Z", "date_end": "2024-01-01T15:00:00Z"}
    ]
    feed.http.get = MagicMock(return_value=response)
    assert data.session_catalog(feed, 2024)[0]["session_key"] == 1
    assert len(feed.memory) == 1
    feed.get("location", {"session_key": 1}, archive=True)
    assert len(feed.memory) == 1
    feed.memory.clear()
    feed.http.get.side_effect = requests.ConnectionError("offline")
    assert data.session_catalog(feed, 2024)[0]["session_key"] == 1


def test_transient_retry_and_safe_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "CACHE", tmp_path)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    feed = data.RaceFeed()
    busy, okay = MagicMock(), MagicMock()
    busy.status_code, busy.headers = 429, {"Retry-After": "0"}
    okay.status_code, okay.json.return_value = 200, []
    feed.http.get = MagicMock(side_effect=[busy, okay])
    assert feed.get("position", {}) == []
    assert feed.http.get.call_count == 2
    failure = MagicMock(status_code=401)
    failure.raise_for_status.side_effect = requests.HTTPError("secret-url", response=failure)
    feed.http.get = MagicMock(return_value=failure)
    with pytest.raises(data.RaceFeedError, match=r"provider unavailable \(401\)") as error:
        feed.get("position", {})
    assert "secret-url" not in str(error.value)


def test_replay_caches_bundle_and_telemetry_uses_race_bounds(tmp_path, monkeypatch, race_records):
    monkeypatch.setattr(data, "CACHE", tmp_path)
    meta, records, locations, _ = race_records
    feed = MagicMock()

    def get(endpoint, *args, **kwargs):
        return locations if endpoint == "location" else records[endpoint]

    feed.get.side_effect = get
    bundle = data.load_replay(feed, meta)
    count = feed.get.call_count
    assert data.load_replay(feed, meta) == bundle
    assert feed.get.call_count == count
    assert not list(tmp_path.glob("*.tmp.json"))
    feed.get.return_value = []
    feed.get.side_effect = None
    assert data.load_telemetry(feed, 1, 1, bundle["origin"], 180) == []


def test_live_window_freshness_and_track_reuse(monkeypatch, race_records):
    meta, records, _locations, _ = race_records
    now = pd.Timestamp(datetime.now(UTC))
    feed = data.RaceFeed()

    def get(endpoint, params, **kwargs):
        if endpoint == "location":
            return [
                {
                    "driver_number": 1,
                    "date": (now - pd.Timedelta(seconds=2)).isoformat(),
                    "x": 4,
                    "y": 5,
                }
            ]
        if endpoint == "car_data":
            return []
        return records[endpoint]

    monkeypatch.setattr(feed, "get", get)
    snap = data.live_snapshot(feed, meta, 1)
    assert 2 <= snap["sample_age_s"] < 5
    feed.live_tracks[1] = [[1, 2], [3, 4]]
    bundle, telemetry, age = data.load_live(feed, meta, 1)
    assert bundle["track"] == feed.live_tracks[1]
    assert data.timestamp(bundle["origin"]) < now
    assert telemetry == []
    assert age is not None
    assert bundle["duration"] > 180


def test_authentication_renews_expired_token_without_browser_credentials(monkeypatch):
    monkeypatch.setenv("OPENF1_USERNAME", "test-user")
    monkeypatch.setenv("OPENF1_PASSWORD", "test-password")
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    feed = data.RaceFeed(token="old-token")
    denied, okay, token = MagicMock(), MagicMock(), MagicMock()
    denied.status_code, okay.status_code = 401, 200
    okay.json.return_value = []
    token.json.return_value = {"access_token": "new-token", "expires_in": 3600}
    feed.http.post = MagicMock(return_value=token)
    feed.http.get = MagicMock(side_effect=[denied, okay])
    assert feed.get("sessions", {"session_key": "latest"}) == []
    assert feed.http.headers["Authorization"] == "Bearer new-token"
    assert feed.http.post.call_args.args == ("https://api.openf1.org/token",)
    assert feed._expires_at > data.time.monotonic()


def test_authentication_failure_is_sanitised(monkeypatch):
    monkeypatch.setenv("OPENF1_USERNAME", "test-user")
    monkeypatch.setenv("OPENF1_PASSWORD", "secret-test-password")
    feed = data.RaceFeed()
    feed.http.post = MagicMock(side_effect=requests.ConnectionError("secret-test-password"))
    with pytest.raises(data.RaceFeedError, match="Live authentication failed") as error:
        feed.get("sessions", {"session_key": "latest"})
    assert "secret-test-password" not in str(error.value)
