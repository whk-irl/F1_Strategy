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


def test_live_window_freshness_and_track_reuse(monkeypatch, race_records, tmp_path):
    monkeypatch.setattr(data, "LIVE_CAPTURE", tmp_path)
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


@pytest.fixture
def prepared_weekend(race_records):
    meta, _, _, _ = race_records
    meta.update({"meeting_key": 77, "session_name": "Race", "year": 2024})
    return {
        "year": 2024,
        "meeting_key": 77,
        "scheduled_laps": 56,
        "race_session": dict(meta),
        "sessions": [],
    }


def test_confirmed_distance_is_never_borrowed_from_another_session(prepared_weekend):
    meta = prepared_weekend["race_session"]
    assert data.confirmed_race_distance(meta, prepared_weekend) == 56
    assert data.confirmed_race_distance({**meta, "session_key": 2}, prepared_weekend) is None
    assert data.confirmed_race_distance({**meta, "meeting_key": 78}, prepared_weekend) is None
    assert (
        data.confirmed_race_distance({**meta, "session_name": "Sprint"}, prepared_weekend) is None
    )
    assert data.confirmed_race_distance(meta, None) is None
    assert data.confirmed_race_distance(meta, {**prepared_weekend, "scheduled_laps": 56.5}) is None


def test_pinned_race_survives_scheduled_end_and_exposes_prestart(prepared_weekend):
    feed = MagicMock()
    planned = prepared_weekend["race_session"]
    for time_value in ("2024-09-01T12:00:00Z", "2024-09-01T16:30:00Z"):
        assert (
            data.select_live_session(feed, prepared_weekend, now=pd.Timestamp(time_value))
            == planned
        )
    feed.get.assert_not_called()
    feed.get.return_value = []
    assert (
        data.select_live_session(feed, prepared_weekend, now=pd.Timestamp("2024-09-02T12:00:00Z"))
        is None
    )
    feed.get.assert_called_once()


def test_bootstrap_track_uses_only_matching_completed_weekend(prepared_weekend):
    meta = prepared_weekend["race_session"]
    track = [[i, i + 10] for i in range(60)]
    feed = MagicMock()
    weekend = {**prepared_weekend, "track": track}
    assert data.load_bootstrap_track(feed, meta, weekend) == track
    assert data.load_bootstrap_track(feed, {**meta, "meeting_key": 2}, weekend) == []
    feed.get.assert_not_called()


def test_live_uses_verified_distance_from_first_lap(
    prepared_weekend, race_records, monkeypatch, tmp_path
):
    monkeypatch.setattr(data, "LIVE_CAPTURE", tmp_path)
    meta, records, locations, _ = race_records
    feed = data.RaceFeed()

    def get(endpoint, *args, **kwargs):
        return (
            locations
            if endpoint == "location"
            else []
            if endpoint == "car_data"
            else records[endpoint]
        )

    monkeypatch.setattr(feed, "get", get)
    feed.live_tracks[1] = [[1, 2], [3, 4]]
    bundle, _, _ = data.load_live(feed, meta, 1, prepared_weekend)
    assert bundle["observed_laps"] == 2
    assert bundle["total_laps"] == 56
    assert bundle["distance_confirmed"] is True
    assert bundle["capture_ok"] is True
    stored = json.loads((tmp_path / "1" / "snapshots.jsonl").read_text())
    assert stored["total_laps"] == 56
    assert len(stored["laps"]["1"]) <= 2


def test_live_waits_for_first_lap_without_failing_connection(race_records, monkeypatch):
    meta, records, _, _ = race_records
    records["laps"] = []
    feed = data.RaceFeed()
    monkeypatch.setattr(feed, "get", lambda endpoint, *args, **kwargs: records[endpoint])
    with pytest.raises(data.RaceFeedPendingError, match="lights out"):
        data.load_live(feed, meta, 1)


def test_free_timing_keeps_missing_gps_unknown_and_gates_freshness(
    prepared_weekend, monkeypatch, tmp_path
):
    monkeypatch.setattr(data, "LIVE_CAPTURE", tmp_path)
    meta = prepared_weekend["race_session"]
    now = pd.Timestamp.now(tz="UTC")
    snapshot = {
        "observed_at": (now - pd.Timedelta(seconds=3)).isoformat(),
        "current_lap": 12,
        "track_status": "1",
        "drivers": [
            {
                "driver_number": 1,
                "name_acronym": "AAA",
                "position": 2,
                "completed_laps": 11,
                "last_lap_duration": 91.2,
                "gap_to_leader": None,
                "interval": None,
                "compound": "MEDIUM",
                "tyre_age": 8,
                "pit_count": 1,
                "observed_at": (now - pd.Timedelta(seconds=3)).isoformat(),
                "gap_to_leader_updated_at": (now - pd.Timedelta(seconds=3)).isoformat(),
                "interval_updated_at": (now - pd.Timedelta(seconds=3)).isoformat(),
                "last_lap_updated_at": (now - pd.Timedelta(seconds=4)).isoformat(),
            }
        ],
        "stints": [
            {"driver_number": 1, "stint_number": 1, "lap_start": 1, "compound": "SOFT"},
            {"driver_number": 1, "stint_number": 2, "lap_start": None, "compound": "MEDIUM"},
        ],
        "weather": {"rainfall": 0},
        "race_control": [],
    }
    feed = data.RaceFeed()
    weekend = {**prepared_weekend, "track": [[i, i] for i in range(60)]}
    bundle, telemetry, age = data.load_free_live(feed, snapshot, meta, 1, weekend)
    assert telemetry == []
    assert bundle["locations"] == {}
    assert bundle["capabilities"]["position"] is False
    assert bundle["track"] == weekend["track"]
    assert bundle["total_laps"] == 56
    assert bundle["laps"]["1"][-1][2] == 12
    assert bundle["latest_completed"]["1"][3] == 91.2
    assert bundle["intervals"]["1"][-1][1:] == [None, None]
    assert {row["compound"] for row in bundle["stints"]} == {"SOFT", "MEDIUM"}
    assert age is not None
    assert 3 <= age < 6
    first_gap_t = bundle["intervals"]["1"][-1][0]
    fresh_other_driver = {**snapshot, "observed_at": now.isoformat()}
    refreshed, _, _ = data.load_free_live(feed, fresh_other_driver, meta, 1, weekend)
    assert refreshed["intervals"]["1"][-1][0] == first_gap_t
    stale = {**snapshot, "observed_at": (now - pd.Timedelta(seconds=90)).isoformat()}
    _, _, stale_age = data.load_free_live(feed, stale, meta, 1, weekend)
    assert stale_age is not None
    assert stale_age >= 90


def test_live_capture_can_be_disabled_or_fail_without_stopping_timing(
    prepared_weekend, monkeypatch, tmp_path
):
    feed = data.RaceFeed()
    bundle = {
        "id": 1,
        "meta": prepared_weekend["race_session"],
        "updated_at": datetime.now(UTC).isoformat(),
        "source": "test timing",
        "origin": prepared_weekend["race_session"]["date_start"],
        "duration": 1,
        "total_laps": 56,
        "drivers": [],
        "laps": {},
        "positions": {},
        "intervals": {},
        "locations": {},
        "stints": [],
        "pits": [],
        "weather": [],
        "messages": [],
    }
    monkeypatch.setenv("PITWALL_LIVE_CAPTURE", "0")
    data.capture_live_snapshot(feed, bundle, [], 1)
    assert bundle["capture_ok"] is None
    monkeypatch.setenv("PITWALL_LIVE_CAPTURE", "1")
    invalid = tmp_path / "not-a-directory"
    invalid.write_text("occupied")
    monkeypatch.setattr(data, "LIVE_CAPTURE", invalid)
    data.capture_live_snapshot(feed, bundle, [], 1)
    assert bundle["capture_ok"] is False


def test_prospective_decision_log_is_structured_and_session_bound(
    prepared_weekend, monkeypatch, tmp_path
):
    monkeypatch.setattr(data, "LIVE_CAPTURE", tmp_path)
    feed = data.RaceFeed()
    decision = {
        "session_key": 1,
        "driver": 1,
        "lap": 12,
        "t": 1020,
        "input_updated_at": datetime.now(UTC).isoformat(),
        "action": "stay_out",
        "source": "free",
        "fit": "HARD",
        "stop_lap": 25,
        "pit_loss_s": 20,
        "rejoin_position": 7,
        "sample_age_s": 2,
        "arbitrary_browser_text": "never-persist-this",
    }
    assert data.record_live_decision(feed, decision, prepared_weekend)
    text = (tmp_path / "1" / "recommendations.jsonl").read_text()
    record = json.loads(text)
    assert record["stop_lap"] == 25
    assert record["action"] == "stay_out"
    assert "never-persist-this" not in text
    for invalid in (
        {**decision, "session_key": 999},
        {**decision, "lap": 57},
        {**decision, "action": ["stay_out"]},
        {**decision, "fit": []},
        {**decision, "input_updated_at": "NaT"},
    ):
        assert not data.record_live_decision(feed, invalid, prepared_weekend)
    assert len((tmp_path / "1" / "recommendations.jsonl").read_text().splitlines()) == 1
