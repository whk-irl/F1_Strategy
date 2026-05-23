"""Thin wrapper around the OpenF1 public REST API.

All methods return plain Python dicts/lists so callers don't depend on any
OpenF1-specific types.  Responses are cached in memory for ``ttl`` seconds
to avoid hammering the API on Streamlit reruns.

The client also handles HTTP 429 (rate-limited) responses by serving the
stale cached value if one exists and entering a global cool-off window so
follow-up calls don't re-hit the API.  This is critical during live races
where the Streamlit live tab makes ~7 endpoint calls per refresh tick.

API base: https://api.openf1.org/v1/
Docs: https://openf1.org/
"""

from __future__ import annotations

import time
from typing import Any, cast

import requests

_BASE = "https://api.openf1.org/v1"
_SESSION_TIMEOUT = 10  # HTTP request timeout (seconds)
_RATE_LIMIT_BACKOFF_S = 30.0  # cool-off window after a 429


class OpenF1Client:
    """Fetch live and historical data from the OpenF1 API.

    Args:
        ttl: Cache TTL in seconds.  Responses older than this are refetched.
            Default 60s — laps are ~90s long during a sprint/race so a 60s
            TTL is fresh enough for strategy decisions while halving the
            request rate compared to the previous 25s default.
    """

    def __init__(self, ttl: int = 60) -> None:
        self._ttl = ttl
        self._cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._session = requests.Session()
        # Monotonic timestamp until which we should not hit the API again.
        self._rate_limited_until: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def is_rate_limited(self) -> bool:
        """Return True if we are currently in a 429 cool-off window."""
        return time.monotonic() < self._rate_limited_until

    def _get(self, endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        key = endpoint + str(sorted(params.items()))
        now = time.monotonic()

        cached = self._cache.get(key)
        if cached is not None:
            ts, data = cached
            if now - ts < self._ttl:
                return data

        # If we're still in a 429 cool-off, serve the (possibly stale) cache
        # rather than hammering the API and getting blocked further.
        if now < self._rate_limited_until and cached is not None:
            return cached[1]

        url = f"{_BASE}/{endpoint}"
        try:
            resp = self._session.get(url, params=params, timeout=_SESSION_TIMEOUT)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                # Respect Retry-After header if present, else use default backoff.
                retry_after = exc.response.headers.get("Retry-After")
                try:
                    backoff = float(retry_after) if retry_after else _RATE_LIMIT_BACKOFF_S
                except ValueError:
                    backoff = _RATE_LIMIT_BACKOFF_S
                self._rate_limited_until = now + backoff
                if cached is not None:
                    # Refresh the cache timestamp so we don't immediately re-try
                    # the next call within the cool-off window.
                    self._cache[key] = (now, cached[1])
                    return cached[1]
            raise

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
