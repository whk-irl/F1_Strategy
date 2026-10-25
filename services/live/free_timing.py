"""Experimental, unauthenticated FastF1 timing recorder for a local race test.

This uses the upstream client's explicit ``no_auth=True`` option. Public timing
availability is session dependent. GPS and car telemetry are deliberately not
requested. A current heartbeat is never treated as a current timing measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, suppress
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from signalrcore.hub_connection_builder import HubConnectionBuilder
from signalrcore.messages.completion_message import CompletionMessage

DEFAULT_RAW_PATH = Path("data/live/free-timing.txt")
DEFAULT_SNAPSHOT_PATH = Path("data/live/free-timing.json")
TOPICS = (
    "Heartbeat",
    "DriverList",
    "RaceControlMessages",
    "SessionInfo",
    "SessionStatus",
    "TimingAppData",
    "TrackStatus",
    "WeatherData",
    "TimingData",
    "LapCount",
)


def _date(value: Any) -> datetime | None:
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result.replace(tzinfo=UTC) if result.tzinfo is None else result.astimezone(UTC)
    except (TypeError, ValueError, OverflowError):
        return None


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number.is_integer() else None


def _seconds(value: Any) -> float | None:
    if isinstance(value, Mapping):
        value = value.get("Value")
    try:
        parts = str(value).split(":")
        if len(parts) == 2:
            return _number(float(parts[0]) * 60 + float(parts[1]))
        return _number(value)
    except (ValueError, TypeError, OverflowError):
        return None


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {str(i): item for i, item in enumerate(value)} if isinstance(value, list) else {}


def _merge(previous: dict[str, Any], update: Mapping[str, Any]) -> None:
    """Preserve omitted fields and indexed list entries in provider deltas."""
    for key, value in update.items():
        if isinstance(value, (dict, list)):
            nested = _mapping(previous.get(key))
            _merge(nested, _mapping(value))
            previous[key] = nested
        else:
            previous[key] = value


def _normal_name(value: Any) -> str:
    return "".join(char for char in str(value).lower() if char.isalnum())


def _session_info(raw: dict[str, Any]) -> dict[str, Any]:
    meeting = _mapping(raw.get("Meeting"))
    circuit = _mapping(meeting.get("Circuit"))
    start, end = _date(raw.get("StartDate")), _date(raw.get("EndDate"))
    # SessionInfo uses local dates and a separate GMT offset for some meetings.
    offset = str(raw.get("GmtOffset", ""))
    raw_start = str(raw.get("StartDate", ""))
    has_timezone = raw_start.endswith("Z") or "+" in raw_start[10:] or "-" in raw_start[10:]
    if offset and start and not has_timezone:
        try:
            sign = -1 if offset.startswith("-") else 1
            hours, minutes, seconds = (int(part) for part in offset.lstrip("+-").split(":"))
            shift = sign * timedelta(hours=hours, minutes=minutes, seconds=seconds)
            start = start - shift
            end = end - shift if end else None
        except ValueError:
            pass
    return {
        "year": start.year if start else None,
        "meeting_name": meeting.get("Name"),
        "circuit_short_name": circuit.get("ShortName", meeting.get("Location")),
        "session_name": raw.get("Name", raw.get("Type")),
        "date_start": start.isoformat() if start else None,
        "date_end": end.isoformat() if end else None,
        "f1_session_key": raw.get("Key"),
    }


class FreeTimingState:
    """Accumulate partial timing topics without fabricating absent measurements."""

    def __init__(self) -> None:
        self.topics: dict[str, dict[str, Any]] = {}
        self.topic_updated_at: dict[str, str] = {}
        self.received_at: str | None = None
        self.connected = False
        self.message_count = 0
        self._stint_starts: dict[tuple[str, str], int] = {}
        self._driver_times: dict[str, dict[str, str]] = {}

    def update(
        self, topic: str, payload: Any, event_at: Any = None, *, now: datetime | None = None
    ) -> bool:
        if topic not in TOPICS:
            return False
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                return False
        if not isinstance(payload, Mapping):
            return False
        event = _date(event_at)
        previous = _date(self.topic_updated_at.get(topic))
        if event and previous and event < previous:
            return False
        if topic == "SessionInfo":
            old = self.topics.get("SessionInfo", {})
            changed_session = any(
                old.get(key) is not None
                and payload.get(key) is not None
                and old[key] != payload[key]
                for key in ("Key", "StartDate")
            )
            if changed_session:
                self.topics.clear()
                self.topic_updated_at.clear()
                self._stint_starts.clear()
                self._driver_times.clear()
                self.message_count = 0
        current = now or datetime.now(UTC)
        self.received_at = current.astimezone(UTC).isoformat()
        meaningful = topic != "TimingData" or any(
            bool(_mapping(line)) for line in _mapping(payload.get("Lines")).values()
        )
        if event and meaningful:
            self.topic_updated_at[topic] = event.isoformat()
        if event and topic in {"TimingData", "TimingAppData"}:
            for driver, line in _mapping(payload.get("Lines")).items():
                line = _mapping(line)
                if not line:
                    continue
                stamps = self._driver_times.setdefault(driver, {})
                if topic == "TimingData":
                    stamps["observed_at"] = event.isoformat()
                    for source, target in (
                        ("GapToLeader", "gap_to_leader_updated_at"),
                        ("IntervalToPositionAhead", "interval_updated_at"),
                        ("LastLapTime", "last_lap_updated_at"),
                    ):
                        if source in line:
                            stamps[target] = event.isoformat()
                elif _mapping(line.get("Stints")):
                    stamps["tyre_updated_at"] = event.isoformat()
        if topic == "TimingAppData" and event:
            old_lines = _mapping(self.topics.get(topic, {}).get("Lines"))
            timing_lines = _mapping(self.topics.get("TimingData", {}).get("Lines"))
            for driver, line in _mapping(payload.get("Lines")).items():
                old_stints = _mapping(_mapping(old_lines.get(driver)).get("Stints"))
                for stint in _mapping(_mapping(line).get("Stints")):
                    if stint not in old_stints:
                        completed = _integer(_mapping(timing_lines.get(driver)).get("NumberOfLaps"))
                        if completed is not None:
                            self._stint_starts[(driver, stint)] = completed + 1
        _merge(self.topics.setdefault(topic, {}), payload)
        self.message_count += 1
        return True

    def snapshot(self) -> dict[str, Any]:
        roster = self.topics.get("DriverList", {})
        lines = _mapping(self.topics.get("TimingData", {}).get("Lines"))
        apps = _mapping(self.topics.get("TimingAppData", {}).get("Lines"))
        drivers, stints = [], []
        for number in sorted(set(roster) | set(lines) | set(apps)):
            if not number.isdigit():
                continue
            person, timing = _mapping(roster.get(number)), _mapping(lines.get(number))
            driver_stints = _mapping(_mapping(apps.get(number)).get("Stints"))
            latest = max(driver_stints, key=lambda value: _integer(value) or 0, default=None)
            tyre = _mapping(driver_stints.get(latest)) if latest is not None else {}
            completed = _integer(timing.get("NumberOfLaps"))
            interval = timing.get("IntervalToPositionAhead")
            if isinstance(interval, Mapping):
                interval = interval.get("Value")
            drivers.append(
                {
                    "driver_number": int(number),
                    "name_acronym": person.get("Tla"),
                    "full_name": person.get("FullName", person.get("BroadcastName")),
                    "team_name": person.get("TeamName"),
                    "team_colour": person.get("TeamColour"),
                    "position": _integer(timing.get("Position")),
                    "lap_number": completed,
                    "completed_laps": completed,
                    "last_lap_duration": _seconds(timing.get("LastLapTime")),
                    "gap_to_leader": timing.get("GapToLeader"),
                    "interval": interval,
                    "compound": tyre.get("Compound"),
                    "tyre_age": _number(tyre.get("TotalLaps")),
                    "pit_count": _integer(timing.get("NumberOfPitStops")),
                    "in_pit": timing.get("InPit"),
                    "retired": timing.get("Retired"),
                    **self._driver_times.get(number, {}),
                }
            )
            for stint_number, stint in driver_stints.items():
                stint = _mapping(stint)
                new = str(stint.get("New", "")).lower() == "true"
                stints.append(
                    {
                        "driver_number": int(number),
                        "stint_number": (_integer(stint_number) or 0) + 1,
                        "compound": stint.get("Compound"),
                        "lap_start": 1
                        if stint_number == "0"
                        else self._stint_starts.get((number, stint_number)),
                        "tyre_age_at_start": 0 if new else None,
                        "tyre_age": _number(stint.get("TotalLaps")),
                    }
                )
        weather = self.topics.get("WeatherData", {})
        weather_values = {
            "air_temperature": _number(weather.get("AirTemp")),
            "track_temperature": _number(weather.get("TrackTemp")),
            "humidity": _number(weather.get("Humidity")),
            "wind_speed": _number(weather.get("WindSpeed")),
            "rainfall": _number(weather.get("Rainfall")),
        }
        messages = []
        for message in _mapping(
            self.topics.get("RaceControlMessages", {}).get("Messages")
        ).values():
            message = _mapping(message)
            messages.append(
                {
                    "date": message.get("Utc"),
                    "message": message.get("Message"),
                    "flag": message.get("Flag"),
                    "category": message.get("Category"),
                }
            )
        lap = self.topics.get("LapCount", {})
        return {
            "schema_version": 1,
            "source": "fastf1_signalr_public",
            "experimental": True,
            "connected": self.connected,
            "message_count": self.message_count,
            "received_at": self.received_at,
            "observed_at": self.topic_updated_at.get("TimingData"),
            "stream_at": self.topic_updated_at.get("Heartbeat"),
            "topic_updated_at": dict(self.topic_updated_at),
            "received_topics": sorted(self.topics),
            "session": _session_info(self.topics.get("SessionInfo", {})),
            "session_status": self.topics.get("SessionStatus", {}).get("Status"),
            "current_lap": _integer(lap.get("CurrentLap")),
            "total_laps": _integer(lap.get("TotalLaps")),
            "track_status": self.topics.get("TrackStatus", {}).get("Status"),
            "track_status_message": self.topics.get("TrackStatus", {}).get("Message"),
            "drivers": drivers,
            "stints": stints,
            "weather": weather_values,
            "race_control": messages,
            "capabilities": {
                "timing": bool(lines),
                "tyres": bool(apps),
                "weather": bool(weather),
                "race_control": "RaceControlMessages" in self.topics,
                "position": False,
                "car_data": False,
            },
        }


def write_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    """Publish a whole JSON document, including on Windows Streamlit readers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(snapshot, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_free_snapshot(
    path: Path = DEFAULT_SNAPSHOT_PATH,
    *,
    expected_meta: Mapping[str, Any] | None = None,
    max_age_s: float | None = 45,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """Read a safe matching session; ``max_age_s=None`` permits stale UI display.

    Native F1 session keys are not compared with unrelated OpenF1 keys. Matching
    requires name, circuit, year and race day when those are supplied by the UI.
    """
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
            return None
        if snapshot.get("source") != "fastf1_signalr_public":
            return None
        session = snapshot.get("session", {})
        if not isinstance(session, dict) or not isinstance(snapshot.get("drivers"), list):
            return None
        if expected_meta:
            for key in ("year", "session_name", "circuit_short_name"):
                if expected_meta.get(key) is not None and (
                    _normal_name(session.get(key)) != _normal_name(expected_meta[key])
                ):
                    return None
            expected_start = _date(expected_meta.get("date_start"))
            actual_start = _date(session.get("date_start"))
            if expected_start and (
                not actual_start or expected_start.date() != actual_start.date()
            ):
                return None
        observed = _date(snapshot.get("observed_at"))
        current = now or datetime.now(UTC)
        age = (current.astimezone(UTC) - observed).total_seconds() if observed else None
        if age is not None and age < -10:
            return None
        if max_age_s is not None and (age is None or age > max_age_s):
            return None
        snapshot["sample_age_s"] = max(0, age) if age is not None else None
        return snapshot
    except (OSError, ValueError, TypeError, OverflowError):
        return None


def _upstream_client() -> Any:
    # The upstream module touches its auth cache at import time even in no-auth
    # mode. Keep this import out of Streamlit snapshot reads and offline checks.
    from fastf1.livetiming.client import SignalRClient

    return SignalRClient


class FreeTimingClient:
    """Record upstream raw messages and expose a small local current-state file."""

    def __init__(
        self,
        raw_path: Path = DEFAULT_RAW_PATH,
        snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
        *,
        probe_seconds: int | None = None,
    ) -> None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = FreeTimingState()
        self.snapshot_path = snapshot_path
        self.probe_seconds = probe_seconds
        self._started = time.monotonic()
        self._files = ExitStack()
        self.connected_once = False
        owner = self
        base = _upstream_client()

        class Recorder(base):  # type: ignore[misc,valid-type]
            def _run(self) -> None:
                # FastF1 3.8.3 passes access_token_factory=None in no-auth mode;
                # signalrcore rejects that option. Omitting the optional auth
                # factory makes the same public request without any credentials.
                self._output_file = owner._files.enter_context(
                    open(self.filename, self.filemode, encoding="utf-8")  # noqa: SIM115
                )
                response = requests.options(self._negotiate_url, headers=self.headers, timeout=10)
                # The load-balancer OPTIONS request may return 405 while still
                # setting its routing cookie; the actual negotiate is POST.
                if response.status_code != 405:
                    response.raise_for_status()
                cookie = response.cookies.get("AWSALBCORS")
                if cookie:
                    self.headers["Cookie"] = f"AWSALBCORS={cookie}"
                self._connection = (
                    HubConnectionBuilder()
                    .with_url(
                        self._connection_url,
                        options={"verify_ssl": True, "headers": self.headers},
                    )
                    .build()
                )
                self._connection.on_open(self._on_connect)
                self._connection.on_close(self._on_close)
                self._connection.on("feed", self._on_message)
                self._connection.start()
                deadline = time.monotonic() + 10
                while not self._is_connected:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Public timing connection did not open")
                    time.sleep(0.1)
                self._connection.send("Subscribe", [self.topics], on_invocation=self._on_message)

            def _on_message(self, msg: Any) -> None:
                super()._on_message(msg)
                owner._consume(msg)

            def _on_connect(self) -> None:
                super()._on_connect()
                owner.state.connected = True
                owner.connected_once = True
                owner._publish()

            def _on_close(self) -> None:
                super()._on_close()
                owner.state.connected = False
                owner._publish()

            def _supervise(self) -> None:
                if owner.probe_seconds is None:
                    while self._is_connected:
                        time.sleep(0.2)
                    return
                deadline = owner._started + owner.probe_seconds
                while time.monotonic() < deadline:
                    time.sleep(min(0.2, max(0, deadline - time.monotonic())))
                self._exit()

        self.client = Recorder(str(raw_path), filemode="a", timeout=0, no_auth=True)
        self.client.topics = list(TOPICS)
        self._publish()

    def _publish(self) -> None:
        write_snapshot(self.snapshot_path, self.state.snapshot())

    def _consume(self, msg: Any) -> None:
        changed = False
        if isinstance(msg, CompletionMessage) and isinstance(msg.result, dict):
            # Session changes clear retained state before the new snapshot merges.
            topics = sorted(msg.result, key=lambda topic: topic != "SessionInfo")
            for topic in topics:
                payload = msg.result[topic]
                changed = self.state.update(topic, payload) or changed
        elif isinstance(msg, list) and len(msg) >= 2:
            changed = self.state.update(msg[0], msg[1], msg[2] if len(msg) > 2 else None)
        if changed:
            self._publish()

    def start(self) -> None:
        attempts = 0
        while True:
            try:
                # Own the interrupt boundary: upstream start() consumes Ctrl+C,
                # which would otherwise look like a disconnect and reconnect.
                self.client._run()
                self.client._supervise()
            except KeyboardInterrupt:
                return
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code in {401, 403}:
                    raise PermissionError("Public provider access denied") from None
                if self.probe_seconds is not None:
                    raise
            except Exception:
                if self.probe_seconds is not None:
                    raise
            finally:
                connection = getattr(self.client, "_connection", None)
                if connection is not None:
                    with suppress(Exception):
                        connection.stop()
                self._files.close()
            if self.probe_seconds is not None:
                return
            attempts += 1
            if attempts >= 5:
                raise ConnectionError("Public timing disconnected after bounded retries")
            time.sleep(min(60, attempts * 10))


def _probe_report(snapshot: dict[str, Any] | None, *, connected: bool = False) -> dict[str, Any]:
    snapshot = snapshot or {}
    age = snapshot.get("sample_age_s")
    fresh = age is not None and 0 <= age <= 45
    timing = bool(snapshot.get("capabilities", {}).get("timing"))
    return {
        "provider": "fastf1_signalr_public",
        "experimental": True,
        "status": "ready" if fresh and timing else "waiting" if connected else "failed",
        "connected": connected,
        "sample_count": snapshot.get("message_count", 0),
        "latest_sample_at": snapshot.get("observed_at"),
        "sample_age_s": age,
        "topics": sorted(snapshot.get("received_topics", [])),
        "capabilities": deepcopy(
            snapshot.get("capabilities", {"position": False, "car_data": False})
        ),
        "notes": [
            "Public timing is session dependent; live access is not guaranteed.",
            "GPS and car telemetry are unavailable in this mode.",
        ],
    }


def probe_free_timing(
    seconds: int = 15,
    *,
    raw_path: Path = DEFAULT_RAW_PATH,
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
) -> dict[str, Any]:
    """Bound the upstream connection (including negotiate) in a child process."""
    if not 1 <= seconds <= 60:
        raise ValueError("Probe duration must be between 1 and 60 seconds")
    command = [
        sys.executable,
        "-m",
        "services.live.free_timing",
        "--worker",
        "--probe-seconds",
        str(seconds),
        "--raw",
        str(raw_path),
        "--snapshot",
        str(snapshot_path),
    ]
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=seconds + 10)
        for line in reversed(process.stdout.splitlines()):
            try:
                report = json.loads(line)
                if isinstance(report, dict) and report.get("provider") == "fastf1_signalr_public":
                    return report
            except ValueError:
                continue
    except (subprocess.TimeoutExpired, OSError):
        pass
    return _probe_report(None)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT_PATH)
    parser.add_argument("--probe-seconds", type=int)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.probe_seconds is not None and not 1 <= args.probe_seconds <= 60:
        parser.error("--probe-seconds must be between 1 and 60")
    if args.probe_seconds and not args.worker:
        report = probe_free_timing(
            args.probe_seconds, raw_path=args.raw, snapshot_path=args.snapshot
        )
    else:
        try:
            client = FreeTimingClient(args.raw, args.snapshot, probe_seconds=args.probe_seconds)
            client.start()
        except Exception as exc:
            # Upstream exception messages can contain URLs/bodies; emit only a safe status.
            report = _probe_report(None)
            report["error_code"] = type(exc).__name__
        else:
            snapshot = load_free_snapshot(args.snapshot, max_age_s=None)
            report = _probe_report(snapshot, connected=client.connected_once)
    print(json.dumps(report, allow_nan=False))
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
