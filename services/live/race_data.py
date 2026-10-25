"""Recorded GPS/timing bundles and bounded live polling for the race center."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, cast
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "race_center"
ARCHIVE = ROOT / "data" / "lake" / "openf1_raw"
LIVE_CAPTURE = ROOT / "data" / "live"


class RaceFeedError(RuntimeError):
    """The data provider could not deliver a usable race feed."""


class RaceFeedPendingError(RaceFeedError):
    """The session is selected, but its first usable timing samples have not arrived."""


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
        self.free_bundles: dict[int, dict[str, Any]] = {}
        self.capture_lock = Lock()

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


def select_live_session(
    feed: RaceFeed,
    weekend: dict[str, Any] | None = None,
    *,
    now: pd.Timestamp | None = None,
) -> dict[str, Any] | None:
    """Prefer the prepared race; retain it through delayed starts and stoppages.

    A pinned upcoming race is returned before lights out so the bridge can show its
    actual start time. A six-hour grace period allows fresh timing to continue past
    the provider's nominal session end without silently switching to another session.
    """
    current = now if now is not None else pd.Timestamp.now(tz="UTC")
    planned = (weekend or {}).get("race_session")
    if isinstance(planned, dict) and planned.get("session_key") and planned.get("date_start"):
        start = timestamp(planned["date_start"])
        end = timestamp(planned.get("date_end", start + pd.Timedelta(hours=3)))
        if start - pd.Timedelta(days=7) <= current <= end + pd.Timedelta(hours=6):
            return planned
    rows = feed.get("sessions", {"session_key": "latest"}, ttl=15)
    return next(
        (
            item
            for item in rows
            if timestamp(item["date_start"])
            <= current
            <= timestamp(item["date_end"]) + pd.Timedelta(hours=2)
        ),
        None,
    )


def confirmed_race_distance(meta: dict[str, Any], weekend: dict[str, Any] | None) -> int | None:
    """Use a verified distance only for the exact prepared meeting and race session."""
    if not weekend or meta.get("session_name") != "Race":
        return None
    planned = weekend.get("race_session") or {}
    if (
        meta.get("session_key") != planned.get("session_key")
        or not meta.get("meeting_key")
        or meta.get("meeting_key") != weekend.get("meeting_key")
        or meta.get("meeting_key") != planned.get("meeting_key")
    ):
        return None
    laps = numeric(weekend.get("scheduled_laps"))
    return int(laps) if laps is not None and laps.is_integer() and 1 <= laps <= 100 else None


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


def load_bootstrap_track(
    feed: RaceFeed, meta: dict[str, Any], weekend: dict[str, Any] | None
) -> list[list[Any]]:
    """Use the prepared weekend's actual practice GPS for the opening race lap."""
    if confirmed_race_distance(meta, weekend) is None or weekend is None:
        return []
    prepared = weekend.get("track")
    if isinstance(prepared, list) and len(prepared) >= 20:
        return cast(list[list[Any]], prepared)
    for session in reversed(weekend.get("sessions", [])):
        session_meta = session.get("meta", {})
        if session_meta.get("meeting_key") != meta.get("meeting_key"):
            continue
        key = session_meta.get("session_key")
        if not key:
            continue
        try:
            laps = feed.get("laps", {"session_key": key}, archive=True)
            candidates = sorted(
                (
                    row
                    for row in laps
                    if row.get("date_start")
                    and not row.get("is_pit_out_lap")
                    and 45 < (numeric(row.get("lap_duration")) or 0) < 220
                ),
                key=lambda row: row["lap_duration"],
            )
            for lap in candidates[:2]:
                start = timestamp(lap["date_start"])
                end = start + pd.Timedelta(seconds=float(lap["lap_duration"]))
                rows = feed.get(
                    "location",
                    {"session_key": key, "driver_number": lap["driver_number"]},
                    after=start.isoformat(),
                    before=end.isoformat(),
                    archive=True,
                )
                points = series_by_driver(rows, start, ["x", "y"], end=end, sample_s=0.6)
                track = [
                    point[1:]
                    for point in points.get(str(lap["driver_number"]), [])
                    if all(numeric(v) is not None for v in point[1:])
                    and (point[1] != 0 or point[2] != 0)
                ]
                if len(track) >= 50:
                    return track
        except (RaceFeedError, ValueError, KeyError):
            continue
    return []


def _sample_age(
    records: list[dict[str, Any]], now: pd.Timestamp, driver: int | None = None
) -> float | None:
    newest = max(
        (
            timestamp(row["date"])
            for row in records
            if row.get("date") and (driver is None or row.get("driver_number") == driver)
        ),
        default=None,
    )
    return max(0.0, (now - newest).total_seconds()) if newest is not None else None


def capture_live_snapshot(
    feed: RaceFeed, bundle: dict[str, Any], telemetry: list[list[Any]], driver: int
) -> None:
    """Append bounded, credential-free shadow-test inputs on the local server."""
    if os.getenv("PITWALL_LIVE_CAPTURE", "1").lower() in ("0", "false", "off"):
        bundle["capture_ok"] = None
        return
    record = {
        "schema_version": 1,
        "updated_at": bundle["updated_at"],
        "meta": bundle["meta"],
        "source": bundle["source"],
        "origin": bundle["origin"],
        "t": bundle["duration"],
        "distance_confirmed": bundle.get("distance_confirmed", False),
        "total_laps": bundle["total_laps"],
        "health": bundle.get("health"),
        "interval_ages": bundle.get("interval_ages"),
        "capabilities": bundle.get("capabilities"),
        "track_status": bundle.get("track_status"),
        "selected_driver": driver,
        "telemetry": telemetry[-1:] if telemetry else [],
        "drivers": bundle["drivers"],
        "laps": {key: rows[-2:] for key, rows in bundle["laps"].items()},
        "latest_completed": bundle.get("latest_completed"),
        "positions": {key: rows[-1:] for key, rows in bundle["positions"].items()},
        "intervals": {key: rows[-1:] for key, rows in bundle["intervals"].items()},
        "locations": {key: rows[-1:] for key, rows in bundle["locations"].items()},
        "stints": bundle["stints"],
        "pits": bundle["pits"][-22:],
        "weather": bundle["weather"][-1:],
        "messages": bundle["messages"][-10:],
    }
    try:
        directory = LIVE_CAPTURE / str(bundle["id"])
        with feed.capture_lock:
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "snapshots.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        bundle["capture_ok"] = True
    except (OSError, ValueError, TypeError):
        bundle["capture_ok"] = False


def record_live_decision(feed: RaceFeed, value: Any, weekend: dict[str, Any] | None) -> bool:
    """Store an allowlisted prospective call for the exact prepared race.

    Browser text, arbitrary paths and extra fields are never persisted. Structured
    numerical fields and small enums make calls reviewable beside the input capture.
    """
    if (
        not isinstance(value, dict)
        or not weekend
        or os.getenv("PITWALL_LIVE_CAPTURE", "1").lower() in ("0", "false", "off")
    ):
        return False
    meta = weekend.get("race_session") or {}
    distance = confirmed_race_distance(meta, weekend)
    if distance is None or value.get("session_key") != meta.get("session_key"):
        return False
    allowed_actions = {
        "unavailable",
        "pit_now",
        "stay_out",
        "run_to_flag",
        "cover",
        "neutralisation",
        "overcut",
        "red_flag",
        "weather_inter",
        "weather_hold",
    }
    action, source = value.get("action"), value.get("source")
    if (
        not isinstance(action, str)
        or action not in allowed_actions
        or not isinstance(source, str)
        or source not in {"free", "openf1"}
    ):
        return False

    def number(field: str, lower: float, upper: float, *, integer: bool = False) -> float | None:
        raw = value.get(field)
        result = numeric(raw)
        if (
            isinstance(raw, bool)
            or result is None
            or not lower <= result <= upper
            or (integer and not result.is_integer())
        ):
            return None
        return result

    driver = number("driver", 1, 99, integer=True)
    lap = number("lap", 1, distance, integer=True)
    t = number("t", 0, 36000)
    if driver is None or lap is None or t is None:
        return False
    roster = {row.get("driver_number") for row in weekend.get("qualifying", [])}
    if roster and int(driver) not in roster:
        return False
    fit = value.get("fit")
    if fit is not None and (
        not isinstance(fit, str)
        or fit not in {"SOFT", "MEDIUM", "HARD", "INTER", "INTERMEDIATE", "WET"}
    ):
        return False
    try:
        input_at = timestamp(value["input_updated_at"])
        if pd.isna(input_at):
            return False
    except (KeyError, ValueError, TypeError):
        return False
    record = {
        "schema_version": 1,
        "recorded_at": datetime.now(UTC).isoformat(),
        "input_updated_at": input_at.isoformat(),
        "session_key": int(meta["session_key"]),
        "meeting_key": meta["meeting_key"],
        "driver": int(driver),
        "lap": int(lap),
        "t": t,
        "action": value["action"],
        "source": value["source"],
        "model": "practice-informed nominal DP v1",
        "fit": fit,
        "stop_lap": number("stop_lap", 1, distance, integer=True),
        "pit_loss_s": number("pit_loss_s", 0, 120),
        "rejoin_position": number("rejoin_position", 1, 30, integer=True),
        "sample_age_s": number("sample_age_s", 0, 86400),
        "weekend_updated_at": weekend.get("updated_at"),
    }
    try:
        with feed.capture_lock:
            directory = LIVE_CAPTURE / str(meta["session_key"])
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "recommendations.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
    except (OSError, TypeError, ValueError):
        return False
    return True


def load_free_live(
    feed: RaceFeed,
    snapshot: dict[str, Any],
    meta: dict[str, Any],
    driver: int,
    weekend: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[list[Any]], float | None]:
    """Adapt recorded public SignalR timing without inventing GPS or telemetry.

    The caller validates the collector's session identity before passing a snapshot.
    Timeline entries are observations of timing changes; no lap-start time is inferred
    from a snapshot received partway through a lap. Available lap times remain explicit.
    """
    now = pd.Timestamp.now(tz="UTC")
    observed = timestamp(snapshot["observed_at"]) if snapshot.get("observed_at") else None
    origin = timestamp(meta["date_start"])
    t = max(0.0, ((observed or now) - origin).total_seconds())
    key = int(meta["session_key"])
    distance = confirmed_race_distance(meta, weekend)
    if key not in feed.live_tracks:
        feed.live_tracks[key] = load_bootstrap_track(feed, meta, weekend)
    if key not in feed.free_bundles:
        feed.free_bundles[key] = {
            "id": key,
            "meta": meta,
            "origin": origin.isoformat(),
            "locations": {},
            "track": feed.live_tracks[key],
            "sectors": [],
            "laps": {},
            "positions": {},
            "intervals": {},
            "latest_completed": {},
            "stints": [],
            "pits": [],
            "weather": [],
            "messages": [],
        }
    bundle = feed.free_bundles[key]
    selected: dict[str, Any] = next(
        (row for row in snapshot.get("drivers", []) if row.get("driver_number") == driver), {}
    )
    selected_time = timestamp(selected["observed_at"]) if selected.get("observed_at") else None
    bundle.update(
        {
            "duration": max(0.0, (now - origin).total_seconds()),
            "total_laps": distance or max(1, int(snapshot.get("current_lap") or 1)),
            "distance_confirmed": distance is not None,
            "observed_laps": int(snapshot.get("current_lap") or 1),
            "source": "Experimental public F1 timing · FastF1 SignalR",
            "capabilities": {"timing": True, "position": False, "car_data": False},
            "updated_at": now.isoformat(),
            "health": {
                "timing_age_s": max(0.0, (now - selected_time).total_seconds())
                if selected_time is not None
                else None,
                "gps_age_s": None,
            },
            "interval_ages": {},
        }
    )
    roster = []
    for row in snapshot.get("drivers", []):
        number = int(row["driver_number"])
        number_key = str(number)
        driver_time = timestamp(row["observed_at"]) if row.get("observed_at") else None
        driver_t = max(0.0, (driver_time - origin).total_seconds()) if driver_time else t
        roster.append(
            {
                field: row.get(field)
                for field in (
                    "driver_number",
                    "name_acronym",
                    "full_name",
                    "team_name",
                    "team_colour",
                )
            }
        )
        completed = int(row.get("completed_laps") or row.get("lap_number") or 0)
        current_lap = min(completed + 1, distance) if distance else completed + 1
        laps = bundle["laps"].setdefault(number_key, [])
        if driver_time is not None and (not laps or laps[-1][2] != current_lap):
            laps.append([driver_t, None, current_lap, None, None, None, None])
        duration = numeric(row.get("last_lap_duration"))
        completion = (
            timestamp(row["last_lap_updated_at"]) if row.get("last_lap_updated_at") else None
        )
        if completed and duration is not None and completion is not None:
            previous = bundle["latest_completed"].get(number_key)
            if previous is None or previous[2] < completed:
                completion_t = max(0.0, (completion - origin).total_seconds())
                bundle["latest_completed"][number_key] = [
                    completion_t,
                    completion_t,
                    completed,
                    duration,
                    None,
                    None,
                    None,
                ]
        position = numeric(row.get("position"))
        if position is not None and driver_time is not None:
            series = bundle["positions"].setdefault(number_key, [])
            if not series or series[-1][0] != driver_t:
                series.append([driver_t, int(position)])
        series = bundle["intervals"].setdefault(number_key, [])
        gap_date = row.get("gap_to_leader_updated_at")
        gap_time = timestamp(gap_date) if gap_date else driver_time if position == 1 else None
        if gap_time is not None:
            gap_t = max(0.0, (gap_time - origin).total_seconds())
            if not series or series[-1][0] < gap_t:
                series.append(
                    [gap_t, 0 if position == 1 else row.get("gap_to_leader"), row.get("interval")]
                )
        interval_date = row.get("interval_updated_at")
        bundle["interval_ages"][number_key] = (
            max(0.0, (now - timestamp(interval_date)).total_seconds()) if interval_date else None
        )
        # A current tyre-age sample supplies a display stint even on a late join.
        # The full collector history below retains known earlier compound usage.
        compound = row.get("compound")
        age = numeric(row.get("tyre_age"))
        if compound:
            current_stint = {
                "driver_number": number,
                "stint_number": int(row.get("pit_count") or 0) + 1,
                "lap_start": current_lap,
                "compound": compound,
                "tyre_age_at_start": age,
                "current_sample": True,
            }
            bundle["stints"] = [
                item
                for item in bundle["stints"]
                if not (item.get("current_sample") and item["driver_number"] == number)
            ] + [current_stint]
    bundle["drivers"] = roster
    known_stints = [
        {**item, "lap_start": item.get("lap_start") or 0} for item in snapshot.get("stints", [])
    ]
    bundle["stints"] = known_stints + [
        item for item in bundle["stints"] if item.get("current_sample")
    ]
    weather = snapshot.get("weather")
    weather_date = snapshot.get("topic_updated_at", {}).get("WeatherData")
    if isinstance(weather, dict) and weather and weather_date:
        weather_time = timestamp(weather_date)
        bundle["weather"] = [{"t": max(0.0, (weather_time - origin).total_seconds()), **weather}]
    bundle["messages"] = _timed(
        snapshot.get("race_control", []),
        origin,
        ["category", "message", "flag", "scope", "driver_number"],
    )
    bundle["track_status"] = snapshot.get("track_status")
    status_date = snapshot.get("topic_updated_at", {}).get("TrackStatus")
    bundle["track_status_t"] = (
        max(0.0, (timestamp(status_date) - origin).total_seconds()) if status_date else None
    )
    bundle["pit_counts"] = {
        str(row["driver_number"]): row.get("pit_count") for row in snapshot.get("drivers", [])
    }
    capture_live_snapshot(feed, bundle, [], driver)
    return bundle, [], max(0.0, (now - observed).total_seconds()) if observed is not None else None


def live_snapshot(feed: RaceFeed, session: dict[str, Any], driver: int) -> dict[str, Any]:
    """Poll bounded current samples; mark freshness from the provider timestamp."""
    now = datetime.now(UTC)
    after = (pd.Timestamp(now) - pd.Timedelta(seconds=45)).isoformat()
    before = pd.Timestamp(now).isoformat()
    key = int(session["session_key"])
    locations = feed.get("location", {"session_key": key}, after=after, before=before, ttl=4)
    telemetry = feed.get(
        "car_data", {"session_key": key, "driver_number": driver}, after=after, before=before, ttl=4
    )
    return {
        "locations": locations,
        "telemetry": telemetry,
        "sample_age_s": _sample_age(locations, pd.Timestamp(now)),
        "driver_age_s": _sample_age(locations, pd.Timestamp(now), driver),
        "updated_at": now.isoformat(),
    }


def load_live(
    feed: RaceFeed,
    meta: dict[str, Any],
    driver: int,
    weekend: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[list[Any]], float | None]:
    """Build a delayed current view, retaining a recorded lap as circuit geometry.

    Current-session data is never archived as an immutable replay. A bounded
    ten-minute location window supplies the first circuit trace; subsequent polls
    request only 45 seconds. The UI uses sample age to expose provider delays.
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
    if not any(row.get("date_start") and row.get("lap_number") == 1 for row in records["laps"]):
        raise RaceFeedPendingError("Waiting for lights out and the first race timing samples.")
    snap = live_snapshot(feed, meta, driver)
    now = timestamp(snap["updated_at"])
    locations = snap["locations"]
    if key not in feed.live_tracks:
        prepared_track = load_bootstrap_track(feed, meta, weekend)
        if prepared_track:
            feed.live_tracks[key] = prepared_track
    if key not in feed.live_tracks:
        locations = feed.get(
            "location",
            {"session_key": key},
            after=(now - pd.Timedelta(minutes=10)).isoformat(),
            before=now.isoformat(),
            ttl=60,
        )
    try:
        bundle = build_bundle(
            meta, records, locations, end_at=now, track_fallback=feed.live_tracks.get(key)
        )
    except RaceFeedError as error:
        if "complete recorded lap" in str(error):
            raise RaceFeedPendingError(
                "Waiting for a complete GPS trace of the circuit."
            ) from error
        raise
    feed.live_tracks[key] = bundle["track"]
    bundle["observed_laps"] = bundle["total_laps"]
    distance = confirmed_race_distance(meta, weekend)
    bundle["distance_confirmed"] = distance is not None
    if distance is not None:
        bundle["total_laps"] = distance
    bundle["source"] = "OpenF1 live positions"
    bundle["capabilities"] = {"timing": True, "position": True, "car_data": True}
    bundle["health"] = {
        "timing_age_s": _sample_age(records["intervals"], now, driver),
        "gps_age_s": snap["driver_age_s"],
        "weather_age_s": _sample_age(records["weather"], now),
    }
    origin = timestamp(bundle["origin"])
    telemetry = series_by_driver(
        snap["telemetry"], origin, ["speed", "throttle", "brake", "n_gear", "drs", "rpm"], end=now
    ).get(str(driver), [])
    capture_live_snapshot(feed, bundle, telemetry, driver)
    return bundle, telemetry, snap["sample_age_s"]
