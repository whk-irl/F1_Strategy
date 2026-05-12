"""Thin wrapper around the OpenF1 public REST API.

All methods return plain Python dicts/lists so callers don't depend on any
OpenF1-specific types.  Responses are cached in memory for ``ttl`` seconds
(default 25 s) to avoid hammering the API on Streamlit reruns.

API base: https://api.openf1.org/v1/
Docs: https://openf1.org/
"""

from __future__ import annotations

import time
from typing import Any, cast

import requests

_BASE = "https://api.openf1.org/v1"
_SESSION_TIMEOUT = 10  # HTTP request timeout (seconds)


class OpenF1Client:
    """Fetch live and historical data from the OpenF1 API.

    Args:
        ttl: Cache TTL in seconds.  Responses older than this are refetched.
    """

    def __init__(self, ttl: int = 25) -> None:
        self._ttl = ttl
        self._cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        key = endpoint + str(sorted(params.items()))
        now = time.monotonic()
        if key in self._cache:
            ts, data = self._cache[key]
            if now - ts < self._ttl:
                return data
        url = f"{_BASE}/{endpoint}"
        resp = self._session.get(url, params=params, timeout=_SESSION_TIMEOUT)
        resp.raise_for_status()
        data = cast(list[dict[str, Any]], resp.json())
        self._cache[key] = (now, data)
        return data

    # ------------------------------------------------------------------
    # Session discovery
    # ------------------------------------------------------------------

    def get_latest_session(self) -> dict[str, Any] | None:
        """Return the most recent session record (race or qualifying)."""
        rows = self._get("sessions", {"session_key": "latest"})
        return rows[0] if rows else None

    def get_sessions(self, year: int) -> list[dict[str, Any]]:
        """Return all sessions for a given season year."""
        return self._get("sessions", {"year": year})

    def get_session(self, session_key: int) -> dict[str, Any] | None:
        """Return a specific session by key."""
        rows = self._get("sessions", {"session_key": session_key})
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    # Lap data
    # ------------------------------------------------------------------

    def get_laps(self, session_key: int, driver_number: int | None = None) -> list[dict[str, Any]]:
        """Return lap records for a session, optionally filtered by driver."""
        params: dict[str, Any] = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number
        return self._get("laps", params)

    def get_latest_lap(self, session_key: int, driver_number: int) -> dict[str, Any] | None:
        """Return the most recently completed lap for a driver."""
        laps = self.get_laps(session_key, driver_number)
        return laps[-1] if laps else None

    # ------------------------------------------------------------------
    # Stints (tyre compound tracking)
    # ------------------------------------------------------------------

    def get_stints(
        self, session_key: int, driver_number: int | None = None
    ) -> list[dict[str, Any]]:
        """Return stint records (compound, lap range) for a session."""
        params: dict[str, Any] = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number
        return self._get("stints", params)

    def get_current_stint(
        self, session_key: int, driver_number: int, current_lap: int
    ) -> dict[str, Any] | None:
        """Return the active stint record for a driver at the given lap."""
        stints = self.get_stints(session_key, driver_number)
        for stint in reversed(stints):
            lap_start = stint.get("lap_start", 0) or 0
            if lap_start <= current_lap:
                return stint
        return None

    # ------------------------------------------------------------------
    # Position
    # ------------------------------------------------------------------

    def get_positions(
        self, session_key: int, driver_number: int | None = None
    ) -> list[dict[str, Any]]:
        """Return position samples for a session."""
        params: dict[str, Any] = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number
        return self._get("position", params)

    def get_latest_position(self, session_key: int, driver_number: int) -> int | None:
        """Return the most recent on-track position for a driver."""
        rows = self.get_positions(session_key, driver_number)
        return int(rows[-1]["position"]) if rows else None

    # ------------------------------------------------------------------
    # Pit stops
    # ------------------------------------------------------------------

    def get_pit_stops(
        self, session_key: int, driver_number: int | None = None
    ) -> list[dict[str, Any]]:
        """Return pit stop records for a session."""
        params: dict[str, Any] = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number
        return self._get("pit", params)

    # ------------------------------------------------------------------
    # Race control (safety car, VSC, flags)
    # ------------------------------------------------------------------

    def get_race_control(self, session_key: int) -> list[dict[str, Any]]:
        """Return race control messages (SC, VSC, red flags, etc.)."""
        return self._get("race_control", {"session_key": session_key})

    def is_safety_car_active(self, session_key: int, current_lap: int) -> bool:
        """Heuristic: true if the most recent SC/VSC message is not 'CLEAR'."""
        msgs = self.get_race_control(session_key)
        sc_msgs = [
            m
            for m in msgs
            if m.get("flag") in ("SAFETY_CAR", "VIRTUAL_SAFETY_CAR")
            or "SAFETY CAR" in m.get("message", "").upper()
        ]
        if not sc_msgs:
            return False
        latest = sc_msgs[-1]
        return "CLEAR" not in latest.get("message", "").upper()

    # ------------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------------

    def get_weather(self, session_key: int) -> list[dict[str, Any]]:
        """Return weather samples for a session."""
        return self._get("weather", {"session_key": session_key})

    def get_latest_weather(self, session_key: int) -> dict[str, Any] | None:
        """Return the most recent weather reading."""
        rows = self.get_weather(session_key)
        return rows[-1] if rows else None

    # ------------------------------------------------------------------
    # Drivers
    # ------------------------------------------------------------------

    def get_drivers(self, session_key: int) -> list[dict[str, Any]]:
        """Return driver roster for a session."""
        return self._get("drivers", {"session_key": session_key})
