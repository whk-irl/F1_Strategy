"""Parse and merge F1 live-timing SignalR payloads.

The official feed at ``livetiming.formula1.com`` sends delta-encoded JSON on
topics such as ``TimingData``, ``TimingAppData``, and ``LapCount``.  This
module maintains merged state and exposes records in the same shape that
:func:`services.live.obs_builder.update_from_openf1` expects from OpenF1.
"""

from __future__ import annotations

import copy
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# TrackStatus codes from the F1 feed (mirrors fastf1.livetiming.data mapping).
_SC_STATUS_CODES = frozenset({"4", "6", "7"})
_ENDED_SESSION_STATUSES = frozenset({"finalised", "finished", "closed", "ends"})


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *update* into *base* (delta frames from SignalR)."""
    for key, val in update.items():
        if key == "_kf":
            continue
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], val)
        else:
            base[key] = val
    return base


def parse_lap_time_s(value: str | float | int | None) -> float | None:
    """Convert F1 timing strings (``'1:23.456'`` or ``'83.210'``) to seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if ":" in text:
        parts = text.split(":")
        try:
            if len(parts) == 2:
                return float(parts[0]) * 60.0 + float(parts[1])
            if len(parts) == 3:
                return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _lines(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    lines = payload.get("Lines")
    return lines if isinstance(lines, dict) else {}


def _driver_line(payload: dict[str, Any] | None, driver_number: int) -> dict[str, Any]:
    line = _lines(payload).get(str(driver_number), {})
    return line if isinstance(line, dict) else {}


def _nested_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _iter_stint_entries(stints_raw: Any) -> list[dict[str, Any]]:
    """Normalize Stints payloads — the free feed uses a list, older feeds use a dict."""
    if isinstance(stints_raw, list):
        return [entry for entry in stints_raw if isinstance(entry, dict)]
    if isinstance(stints_raw, dict):
        return [entry for entry in stints_raw.values() if isinstance(entry, dict)]
    return []


def _stint_lap_start(stint_data: dict[str, Any]) -> int:
    """Return 1-based lap number when a stint started."""
    start_raw = stint_data.get("Start")
    if start_raw is None:
        start_raw = stint_data.get("StartLaps")
    lap_start = _nested_int(start_raw, 0)
    return 1 if lap_start <= 0 else lap_start


def _last_lap_time_s(line: dict[str, Any]) -> float | None:
    last = line.get("LastLapTime")
    if isinstance(last, dict):
        return parse_lap_time_s(last.get("Value"))
    return parse_lap_time_s(last)


@dataclass
class TimingSnapshot:
    """Merged live-timing state across all subscribed SignalR topics."""

    timing_data: dict[str, Any] = field(default_factory=dict)
    timing_app_data: dict[str, Any] = field(default_factory=dict)
    lap_count: dict[str, Any] = field(default_factory=dict)
    session_info: dict[str, Any] = field(default_factory=dict)
    driver_list: dict[str, Any] = field(default_factory=dict)
    track_status: dict[str, Any] = field(default_factory=dict)
    weather_data: dict[str, Any] = field(default_factory=dict)
    race_control_messages: list[dict[str, Any]] = field(default_factory=list)
    connected: bool = False
    has_timing_data: bool = False
    last_update: float = 0.0

    def copy(self) -> TimingSnapshot:
        return TimingSnapshot(
            timing_data=copy.deepcopy(self.timing_data),
            timing_app_data=copy.deepcopy(self.timing_app_data),
            lap_count=copy.deepcopy(self.lap_count),
            session_info=copy.deepcopy(self.session_info),
            driver_list=copy.deepcopy(self.driver_list),
            track_status=copy.deepcopy(self.track_status),
            weather_data=copy.deepcopy(self.weather_data),
            race_control_messages=copy.deepcopy(self.race_control_messages),
            connected=self.connected,
            has_timing_data=self.has_timing_data,
            last_update=self.last_update,
        )

    def apply_topic(self, topic: str, payload: dict[str, Any]) -> None:
        """Merge one SignalR topic update into this snapshot."""
        if topic == "TimingData":
            deep_merge(self.timing_data, payload)
            if _lines(payload) or self.timing_data.get("Lines"):
                self.has_timing_data = True
        elif topic == "TimingAppData":
            deep_merge(self.timing_app_data, payload)
        elif topic == "LapCount":
            deep_merge(self.lap_count, payload)
        elif topic == "SessionInfo":
            deep_merge(self.session_info, payload)
        elif topic == "DriverList":
            deep_merge(self.driver_list, payload)
        elif topic == "TrackStatus":
            deep_merge(self.track_status, payload)
        elif topic == "WeatherData":
            deep_merge(self.weather_data, payload)
        elif topic == "RaceControlMessages":
            msgs = payload.get("Messages")
            if isinstance(msgs, list):
                self.race_control_messages.extend(msgs)
            elif isinstance(msgs, dict):
                self.race_control_messages.extend(msgs.values())

    def current_lap(self, driver_number: int) -> int:
        line = _driver_line(self.timing_data, driver_number)
        laps = _nested_int(line.get("NumberOfLaps"), 0)
        if laps:
            return laps
        return _nested_int(self.lap_count.get("CurrentLap"), 0)

    def total_laps(self) -> int:
        total = _nested_int(self.lap_count.get("TotalLaps"), 0)
        if total:
            return total
        meeting = self.session_info.get("Meeting")
        if isinstance(meeting, dict):
            return _nested_int(meeting.get("Laps"), 0)
        return 0

    def session_key(self) -> int | None:
        meeting = self.session_info.get("Meeting")
        if isinstance(meeting, dict):
            key = meeting.get("Key")
            if key is not None:
                return int(key)
        return None

    def session_meta(self) -> dict[str, Any] | None:
        """Return OpenF1-compatible session metadata, or None if unknown."""
        meeting = self.session_info.get("Meeting")
        if not isinstance(meeting, dict):
            return None
        session_key = self.session_info.get("Key")
        meeting_key = meeting.get("Key")
        if session_key is None and meeting_key is None:
            return None

        circuit = meeting.get("Circuit")
        circuit_short = circuit.get("ShortName") if isinstance(circuit, dict) else None
        country = meeting.get("Country")
        country_name = country.get("Name") if isinstance(country, dict) else None

        session_name = self.session_info.get("Name") or self.session_info.get("Type") or "Session"
        date_start = self.session_info.get("StartDate") or meeting.get("StartDate")

        year = datetime.now(timezone.utc).year
        if date_start:
            with contextlib.suppress(ValueError, TypeError):
                year = int(str(date_start)[:4])

        return {
            "session_key": int(session_key if session_key is not None else meeting_key),
            "meeting_key": int(meeting_key) if meeting_key is not None else None,
            "meeting_name": meeting.get("Name"),
            "country_name": country_name,
            "circuit_short_name": circuit_short,
            "session_name": session_name,
            "session_type": "Race"
            if str(session_name).lower() in ("race", "sprint")
            else session_name,
            "session_status": self.session_info.get("SessionStatus"),
            "total_laps": self.total_laps(),
            "date_start": date_start,
            "date_end": self.session_info.get("EndDate"),
            "year": year,
        }

    def is_session_finalised(self) -> bool:
        """Return True when the feed marks the current session as ended."""
        status = str(self.session_info.get("SessionStatus", "")).lower()
        if status in _ENDED_SESSION_STATUSES:
            return True
        archive = self.session_info.get("ArchiveStatus")
        if isinstance(archive, dict):
            archive_status = str(archive.get("Status", "")).lower()
            if archive_status in {"complete", "finalised", "finished"}:
                return True
        return False

    def drivers(self) -> list[dict[str, Any]]:
        """Return driver roster in OpenF1-compatible shape."""
        roster: list[dict[str, Any]] = []
        for num_str, info in self.driver_list.items():
            if not isinstance(info, dict):
                continue
            try:
                num = int(info.get("RacingNumber", num_str))
            except (TypeError, ValueError):
                continue
            roster.append(
                {
                    "driver_number": num,
                    "name_acronym": info.get("Tla") or info.get("BroadcastName") or "",
                    "team_name": info.get("TeamName") or "",
                }
            )
        if roster:
            return sorted(roster, key=lambda d: d["driver_number"])

        # Fallback: infer from TimingData lines before DriverList arrives.
        for num_str, line in _lines(self.timing_data).items():
            if not isinstance(line, dict):
                continue
            try:
                num = int(num_str)
            except ValueError:
                continue
            roster.append(
                {
                    "driver_number": num,
                    "name_acronym": line.get("Tla") or "",
                    "team_name": line.get("TeamName") or "",
                }
            )
        return sorted(roster, key=lambda d: d["driver_number"])

    def latest_lap(self, driver_number: int) -> dict[str, Any] | None:
        line = _driver_line(self.timing_data, driver_number)
        if not line:
            return None
        lap_number = self.current_lap(driver_number)
        if lap_number <= 0:
            return None
        lap_duration = _last_lap_time_s(line)
        record: dict[str, Any] = {"lap_number": lap_number}
        if lap_duration is not None:
            record["lap_duration"] = lap_duration
        return record

    def latest_position(self, driver_number: int) -> int | None:
        line = _driver_line(self.timing_data, driver_number)
        pos = line.get("Position")
        if pos is None:
            return None
        return _nested_int(pos, 0) or None

    def pit_stop_count(self, driver_number: int) -> int:
        line = _driver_line(self.timing_data, driver_number)
        if line.get("NumberOfPitStops") is not None:
            return _nested_int(line.get("NumberOfPitStops"), 0)
        # Free feed omits NumberOfPitStops — infer from stint count.
        stints = self.stints_for_driver(driver_number)
        return max(len(stints) - 1, 0)

    def stints_for_driver(self, driver_number: int) -> list[dict[str, Any]]:
        line = _driver_line(self.timing_app_data, driver_number)
        entries = _iter_stint_entries(line.get("Stints"))

        stints: list[dict[str, Any]] = []
        for stint_data in entries:
            compound = stint_data.get("Compound")
            lap_start = _stint_lap_start(stint_data)
            total = _nested_int(stint_data.get("TotalLaps") or stint_data.get("Laps"), 0)
            lap_end = lap_start + total - 1 if total > 0 else None
            stints.append(
                {
                    "driver_number": driver_number,
                    "compound": compound,
                    "lap_start": lap_start,
                    "lap_end": lap_end,
                }
            )
        stints.sort(key=lambda s: s["lap_start"])
        return stints

    def all_stints(self) -> list[dict[str, Any]]:
        numbers = {d["driver_number"] for d in self.drivers()}
        for num_str in _lines(self.timing_app_data):
            with contextlib.suppress(ValueError):
                numbers.add(int(num_str))
        result: list[dict[str, Any]] = []
        for num in sorted(numbers):
            result.extend(self.stints_for_driver(num))
        return result

    def current_stint(self, driver_number: int, current_lap: int) -> dict[str, Any] | None:
        stints = self.stints_for_driver(driver_number)
        for stint in reversed(stints):
            if stint["lap_start"] <= current_lap:
                return stint
        return None

    def latest_weather(self) -> dict[str, Any] | None:
        if not self.weather_data:
            return None
        out: dict[str, Any] = {}
        track_temp = self.weather_data.get("TrackTemp")
        if track_temp is not None:
            with contextlib.suppress(TypeError, ValueError):
                out["track_temperature"] = float(track_temp)
        rainfall = self.weather_data.get("Rainfall")
        if rainfall is not None:
            with contextlib.suppress(TypeError, ValueError):
                out["rainfall"] = float(rainfall)
        return out or None

    def is_safety_car_active(self) -> bool:
        status = str(self.track_status.get("Status", "1"))
        if status in _SC_STATUS_CODES:
            return True
        for msg in reversed(self.race_control_messages):
            if not isinstance(msg, dict):
                continue
            message = str(msg.get("Message", "")).upper()
            if "SAFETY CAR" in message or "VIRTUAL SAFETY CAR" in message:
                return "CLEAR" not in message and "ENDED" not in message
        return False
