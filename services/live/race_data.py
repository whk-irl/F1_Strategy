"""Recorded GPS/timing bundles and bounded live polling for the race center."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "race_center"
ARCHIVE = ROOT / "data" / "lake" / "openf1_raw"


class RaceFeedError(RuntimeError):
    """The data provider could not deliver a usable race feed."""


def filtered_url(
    endpoint: str, params: dict[str, Any], after: str | None = None, before: str | None = None
) -> str:
    """Encode OpenF1 comparison expressions without adding an extra equals sign."""
    parts = [urlencode(params)]
    if after:
        parts.append(quote(f"date>{after}", safe=":"))
    if before:
        parts.append(quote(f"date<{before}", safe=":"))
    return f"https://api.openf1.org/v1/{endpoint}?{'&'.join(parts)}"


class RaceFeed:
    """Server-side HTTP client; credentials are never sent to the browser component."""

    def __init__(self, token: str | None = None) -> None:
        self.http = requests.Session()
        token = token or os.getenv("OPENF1_TOKEN")
        self._username = os.getenv("OPENF1_USERNAME")
        self._password = os.getenv("OPENF1_PASSWORD")
        self._can_refresh = bool(self._username and self._password)
        self._expires_at = 0.0 if self._can_refresh and not token else float("inf")
        if token:
            self.http.headers["Authorization"] = f"Bearer {token}"
        self.authenticated = bool(token) or self._can_refresh
        self.last_request = 0.0
        self.memory: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self.live_tracks: dict[int, list[list[Any]]] = {}

    def _authenticate(self) -> None:
        """Renew short-lived tokens on the server without exposing provider responses."""
        try:
            response = self.http.post(
                "https://api.openf1.org/token",
                data={"username": self._username, "password": self._password},
                timeout=20,
            )
            response.raise_for_status()
            body = response.json()
            token = body.get("access_token")
            if not isinstance(token, str) or not token:
                raise ValueError("No access token")
            lifetime = max(60.0, min(float(body.get("expires_in", 3600)), 3600.0))
            self.http.headers["Authorization"] = f"Bearer {token}"
            self._expires_at = time.monotonic() + lifetime - 30
        except (requests.RequestException, ValueError, TypeError) as error:
            raise RaceFeedError(
                "Live authentication failed; check server OpenF1 credentials"
            ) from error

    def get(
        self,
        endpoint: str,
        params: dict[str, Any],
        *,
        after: str | None = None,
        before: str | None = None,
        ttl: float = 0,
        archive: bool = False,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch a public resource with persistent archive caching and transient retries."""
        url = filtered_url(endpoint, params, after, before)
        if ttl and url in self.memory and time.monotonic() < self.memory[url][0]:
            return self.memory[url][1]
        digest = hashlib.sha256(url.encode()).hexdigest()[:24]
        path = CACHE / "responses" / f"{endpoint}-{digest}.json"
        if archive and path.exists() and not refresh:
            return cast(list[dict[str, Any]], json.loads(path.read_text(encoding="utf-8")))
        suffix = "_".join(f"{key}-{value}" for key, value in sorted(params.items()))
        legacy = ARCHIVE / f"{endpoint}_{suffix}.json"
        if archive and not refresh and not after and not before and legacy.exists():
            return cast(list[dict[str, Any]], json.loads(legacy.read_text(encoding="utf-8")))
        for attempt in range(3):
            if self._can_refresh and time.monotonic() >= self._expires_at:
                self._authenticate()
            time.sleep(
                max(
                    0, (0.4 if self.authenticated else 2.1) - (time.monotonic() - self.last_request)
                )
            )
            self.last_request = time.monotonic()
            try:
                response = self.http.get(url, timeout=45)
                if response.status_code == 401 and self._can_refresh and attempt < 2:
                    self._authenticate()
                    continue
                if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(min(float(response.headers.get("Retry-After", 3 * 2**attempt)), 30))
                    continue
                response.raise_for_status()
                records = response.json()
                if not isinstance(records, list):
                    raise RaceFeedError("Unexpected provider response")
                if archive and records:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".tmp.json")
                    temporary.write_text(
                        json.dumps(records, separators=(",", ":")), encoding="utf-8"
                    )
                    temporary.replace(path)
                if ttl:
                    now = time.monotonic()
                    self.memory = {
                        key: value for key, value in self.memory.items() if value[0] > now
                    }
                    self.memory[url] = (now + ttl, records)
                return cast(list[dict[str, Any]], records)
            except requests.RequestException as error:
                if isinstance(error, requests.HTTPError) or attempt == 2:
                    status = getattr(error.response, "status_code", "network")
                    raise RaceFeedError(f"{endpoint}: provider unavailable ({status})") from error
        raise RaceFeedError(f"{endpoint}: retry limit reached")


def session_catalog(feed: RaceFeed, year: int) -> list[dict[str, Any]]:
    """Return finished, non-cancelled races in championship order."""
    params = {"session_name": "Race", "year": year}
    try:
        rows = feed.get("sessions", params, ttl=3600, archive=True, refresh=True)
    except RaceFeedError:
        rows = feed.get("sessions", params, archive=True)
    now = pd.Timestamp.now(tz="UTC")
    finished = [
        row
        for row in rows
        if not row.get("is_cancelled", False) and pd.Timestamp(row["date_end"]) < now
    ]
    finished.sort(key=lambda item: item["date_start"])
    return [{**row, "round": idx + 1} for idx, row in enumerate(finished)]


def timestamp(value: Any) -> pd.Timestamp:
    """Parse a provider timestamp consistently, including mixed fractional precision."""
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def numeric(value: Any) -> float | None:
    """Keep absent or non-finite telemetry absent in the JSON payload."""
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def series_by_driver(
    records: list[dict[str, Any]],
    origin: pd.Timestamp,
    columns: list[str],
    *,
    end: pd.Timestamp | None = None,
    sample_s: float | None = None,
) -> dict[str, list[list[Any]]]:
    """Sort, bound and optionally downsample time series without filling missing data."""
    if not records:
        return {}
    frame = pd.DataFrame(records)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, format="mixed", errors="coerce")
    frame = frame[frame["date"].ge(origin)]
    if end is not None:
        frame = frame[frame["date"].le(end)]
    frame = frame.sort_values("date").drop_duplicates(["driver_number", "date"])
    frame["t"] = (frame["date"] - origin).dt.total_seconds().round(3)
    if sample_s:
        frame["bucket"] = np.floor(frame["t"] / sample_s)
        frame = frame.drop_duplicates(["driver_number", "bucket"], keep="last")
    output = {}
    for driver, group in frame.groupby("driver_number"):
        output[str(int(cast(Any, driver)))] = json.loads(
            group[["t", *columns]].to_json(orient="values")
        )
    return output


def _timed(
    records: list[dict[str, Any]], origin: pd.Timestamp, fields: list[str]
) -> list[dict[str, Any]]:
    rows = []
    for row in records:
        if not row.get("date"):
            continue
        t = (timestamp(row["date"]) - origin).total_seconds()
        if t >= 0:
            rows.append({"t": round(t, 3), **{key: row.get(key) for key in fields}})
    return sorted(rows, key=lambda row: row["t"])


def build_bundle(
    meta: dict[str, Any],
    records: dict[str, list[dict[str, Any]]],
    locations: list[dict[str, Any]],
    *,
    end_at: pd.Timestamp | None = None,
    track_fallback: list[list[Any]] | None = None,
) -> dict[str, Any]:
    """Build a compact race timeline from recorded events and actual GPS samples."""
    laps = records["laps"]
    starts = [
        timestamp(row["date_start"])
        for row in laps
        if row.get("date_start") and row["lap_number"] == 1
    ]
    if not starts:
        raise RaceFeedError("No recorded race start is available")
    origin = min(starts)
    finishes = [
        timestamp(row["date_start"]) + pd.Timedelta(seconds=float(row["lap_duration"]))
        for row in laps
        if row.get("date_start") and numeric(row.get("lap_duration")) is not None
    ]
    end = (
        end_at if end_at is not None else max(finishes) if finishes else timestamp(meta["date_end"])
    )
    positions = series_by_driver(locations, origin, ["x", "y"], end=end, sample_s=0.6)
    roster = {str(row["driver_number"]) for row in records["drivers"]}
    positions = {key: value for key, value in positions.items() if key in roster}
    for driver, points in positions.items():
        positions[driver] = [
            point
            for point in points
            if point[1] is not None and point[2] is not None and (point[1] != 0 or point[2] != 0)
        ]
    clean_laps = sorted(
        [
            row
            for row in laps
            if row.get("date_start")
            and not row.get("is_pit_out_lap")
            and 45 < (numeric(row.get("lap_duration")) or 0) < 220
        ],
        key=lambda row: row["lap_duration"],
    )
    track = track_fallback or []
    sectors = []
    for lap in clean_laps:
        if track:
            break
        start_t = (timestamp(lap["date_start"]) - origin).total_seconds()
        finish_t = start_t + float(lap["lap_duration"])
        candidate = [
            point[1:]
            for point in positions.get(str(lap["driver_number"]), [])
            if start_t <= point[0] <= finish_t
        ]
        if len(candidate) >= 50:
            track = candidate
            sector_time = start_t
            for i in (1, 2):
                duration = numeric(lap.get(f"duration_sector_{i}"))
                if duration is None:
                    break
                sector_time += duration
                point = min(
                    positions[str(lap["driver_number"])],
                    key=lambda value: abs(value[0] - sector_time),
                )
                sectors.append({"label": f"S{i}", "point": point[1:]})
            break
    if not track:
        raise RaceFeedError("No complete recorded lap is available for the circuit map")
    lap_series: dict[str, list[list[Any]]] = {}
    for row in laps:
        if not row.get("date_start"):
            continue
        start_t = (timestamp(row["date_start"]) - origin).total_seconds()
        duration = numeric(row.get("lap_duration"))
        lap_series.setdefault(str(row["driver_number"]), []).append(
            [
                round(start_t, 3),
                round(start_t + duration, 3) if duration else None,
                int(row["lap_number"]),
                duration,
                *[numeric(row.get(f"duration_sector_{i}")) for i in (1, 2, 3)],
            ]
        )
    for rows in lap_series.values():
        rows.sort(key=lambda row: row[0])
    return {
        "id": int(meta["session_key"]),
        "meta": meta,
        "origin": origin.isoformat(),
        "duration": (end - origin).total_seconds(),
        "total_laps": max(int(row["lap_number"]) for row in laps),
        "drivers": [
            {
                key: row.get(key)
                for key in (
                    "driver_number",
                    "name_acronym",
                    "full_name",
                    "team_name",
                    "team_colour",
                )
            }
            for row in records["drivers"]
        ],
        "locations": positions,
        "track": track,
        "sectors": sectors,
        "laps": lap_series,
        "positions": series_by_driver(records["position"], origin, ["position"], end=end),
        "intervals": series_by_driver(
            records.get("intervals", []), origin, ["gap_to_leader", "interval"], end=end
        ),
        "stints": records["stints"],
        "pits": _timed(
            records["pit"],
            origin,
            ["driver_number", "lap_number", "pit_duration", "lane_duration", "stop_duration"],
        ),
        "weather": _timed(
            records["weather"],
            origin,
            ["air_temperature", "track_temperature", "humidity", "rainfall", "wind_speed"],
        ),
        "messages": _timed(
            records["race_control"],
            origin,
            ["category", "message", "flag", "scope", "driver_number"],
        ),
        "source": "OpenF1 recorded positions",
        "updated_at": datetime.now(UTC).isoformat(),
    }


def load_replay(feed: RaceFeed, meta: dict[str, Any]) -> dict[str, Any]:
    """Fetch a race once; bounded GPS requests keep responses below provider limits."""
    key = int(meta["session_key"])
    bundle_path = CACHE / f"race-{key}-v3.json"
    if bundle_path.exists():
        return cast(dict[str, Any], json.loads(bundle_path.read_text(encoding="utf-8")))
    endpoints = (
        "drivers",
        "laps",
        "position",
        "intervals",
        "stints",
        "pit",
        "weather",
        "race_control",
    )
    records = {
        endpoint: feed.get(endpoint, {"session_key": key}, archive=True) for endpoint in endpoints
    }
    starts = [
        timestamp(row["date_start"])
        for row in records["laps"]
        if row.get("date_start") and row["lap_number"] == 1
    ]
    if not starts:
        raise RaceFeedError("Race has no recorded lap start")
    start = min(starts)
    ends = [
        timestamp(row["date_start"]) + pd.Timedelta(seconds=float(row["lap_duration"]))
        for row in records["laps"]
        if row.get("date_start") and numeric(row.get("lap_duration")) is not None
    ]
    end = max(ends)
    locations = []
    cursor = start - pd.Timedelta(seconds=1)
    while cursor < end:
        until = min(cursor + pd.Timedelta(minutes=8), end + pd.Timedelta(seconds=1))
        locations.extend(
            feed.get(
                "location",
                {"session_key": key},
                after=cursor.isoformat(),
                before=until.isoformat(),
                archive=True,
            )
        )
        cursor = until - pd.Timedelta(milliseconds=100) if until < end else end
    bundle = build_bundle(meta, records, locations)
    CACHE.mkdir(parents=True, exist_ok=True)
    temporary = bundle_path.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(bundle, separators=(",", ":"), allow_nan=False), encoding="utf-8"
    )
    temporary.replace(bundle_path)
    return bundle


def load_telemetry(
    feed: RaceFeed, session_key: int, driver_number: int, origin: str, end: float
) -> list[list[Any]]:
    """Fetch one selected car, preserving real speed, pedals, gear, DRS and RPM."""
    rows = feed.get(
        "car_data", {"session_key": session_key, "driver_number": driver_number}, archive=True
    )
    first = timestamp(origin)
    return series_by_driver(
        rows,
        first,
        ["speed", "throttle", "brake", "n_gear", "drs", "rpm"],
        end=first + pd.Timedelta(seconds=end),
        sample_s=0.4,
    ).get(str(driver_number), [])


def live_snapshot(feed: RaceFeed, session: dict[str, Any], driver: int) -> dict[str, Any]:
    """Poll bounded current samples; mark freshness from the provider timestamp."""
    now = datetime.now(UTC)
    after = (pd.Timestamp(now) - pd.Timedelta(seconds=25)).isoformat()
    before = pd.Timestamp(now).isoformat()
    key = int(session["session_key"])
    locations = feed.get("location", {"session_key": key}, after=after, before=before, ttl=4)
    telemetry = feed.get(
        "car_data", {"session_key": key, "driver_number": driver}, after=after, before=before, ttl=4
    )
    newest = max((timestamp(row["date"]) for row in locations), default=None)
    return {
        "locations": locations,
        "telemetry": telemetry,
        "sample_age_s": (pd.Timestamp(now) - newest).total_seconds()
        if newest is not None
        else None,
        "updated_at": now.isoformat(),
    }


def load_live(
    feed: RaceFeed, meta: dict[str, Any], driver: int
) -> tuple[dict[str, Any], list[list[Any]], float | None]:
    """Build a delayed current view, retaining a recorded lap as circuit geometry.

    Current-session data is never archived as an immutable replay. A bounded
    ten-minute location window supplies the first circuit trace; subsequent polls
    request only 25 seconds. The UI uses sample age to expose provider delays.
    """
    key = int(meta["session_key"])
    endpoints = (
        "drivers",
        "laps",
        "position",
        "intervals",
        "stints",
        "pit",
        "weather",
        "race_control",
    )
    records = {endpoint: feed.get(endpoint, {"session_key": key}, ttl=10) for endpoint in endpoints}
    snap = live_snapshot(feed, meta, driver)
    now = timestamp(snap["updated_at"])
    locations = snap["locations"]
    if key not in feed.live_tracks:
        locations = feed.get(
            "location",
            {"session_key": key},
            after=(now - pd.Timedelta(minutes=10)).isoformat(),
            before=now.isoformat(),
            ttl=60,
        )
    bundle = build_bundle(
        meta, records, locations, end_at=now, track_fallback=feed.live_tracks.get(key)
    )
    feed.live_tracks[key] = bundle["track"]
    origin = timestamp(bundle["origin"])
    telemetry = series_by_driver(
        snap["telemetry"], origin, ["speed", "throttle", "brake", "n_gear", "drs", "rpm"], end=now
    ).get(str(driver), [])
    return bundle, telemetry, snap["sample_age_s"]
