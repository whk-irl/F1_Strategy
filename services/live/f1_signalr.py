"""F1 official live timing client via SignalR.

Connects to the same feed that powers
https://www.formula1.com/en/timing/f1-live and exposes an OpenF1-compatible
interface for the Streamlit live tab.

Uses short-lived synchronous connections (connect → subscribe → collect →
disconnect) so each Streamlit rerun can fetch fresh data.  Background threads
are unreliable on Streamlit Cloud.

API: ``wss://livetiming.formula1.com/signalrcore``
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import requests
from signalrcore.hub_connection_builder import HubConnectionBuilder
from signalrcore.messages.completion_message import CompletionMessage

from services.live.timing_state import TimingSnapshot

logger = logging.getLogger(__name__)

_NEGOTIATE_URL = "https://livetiming.formula1.com/signalrcore/negotiate"
_CONNECTION_URL = "wss://livetiming.formula1.com/signalrcore"
_CONNECT_TIMEOUT_S = 15.0
_DEFAULT_COLLECT_S = 4.0

_TOPICS = [
    "Heartbeat",
    "DriverList",
    "ExtrapolatedClock",
    "RaceControlMessages",
    "SessionInfo",
    "SessionStatus",
    "TimingAppData",
    "TimingStats",
    "TrackStatus",
    "WeatherData",
    "Position.z",
    "LapCount",
    "TimingData",
    "TopThree",
]


def resolve_subscription_token(explicit: str | None = None) -> str:
    """Return an F1 subscription token from env, explicit arg, or FastF1 cache."""
    if explicit:
        return explicit.strip()
    env_token = os.getenv("PITWALL_F1_SUBSCRIPTION_TOKEN", "").strip()
    if env_token:
        return env_token
    try:
        import platformdirs

        auth_file = Path(platformdirs.user_data_dir("fastf1")) / "f1auth.json"
        if auth_file.exists():
            return auth_file.read_text(encoding="utf-8").strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


class F1LiveTimingClient:
    """Client for the official F1 live timing SignalR feed.

    Call :meth:`refresh` on each Streamlit rerun to open a short-lived
    WebSocket, merge delta updates into the cached snapshot, and disconnect.
    Public methods mirror :class:`services.live.openf1_api.OpenF1Client`.

    Args:
        subscription_token: Optional F1TV subscription JWT.  When empty the
            client uses the same free feed as
            https://www.formula1.com/en/timing/f1-live.
    """

    def __init__(self, subscription_token: str | None = None) -> None:
        self._token = resolve_subscription_token(subscription_token)
        self._snapshot = TimingSnapshot()
        self._lock = threading.Lock()
        self._error: str | None = None
        self._last_refresh: float = 0.0

    # ------------------------------------------------------------------
    # Connection lifecycle (sync — Streamlit-safe)
    # ------------------------------------------------------------------

    def refresh(self, timeout: float = _DEFAULT_COLLECT_S) -> None:
        """Connect, collect live updates for *timeout* seconds, then disconnect."""
        try:
            self._connect_and_collect(timeout)
            self._error = None
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            logger.warning("F1 SignalR refresh failed: %s", exc)
            with self._lock:
                self._snapshot.connected = False

    def ensure_started(self) -> None:
        """No-op kept for API compatibility — call :meth:`refresh` instead."""

    def _connect_and_collect(self, collect_s: float) -> None:
        headers: dict[str, str] = {}
        r = requests.post(
            _NEGOTIATE_URL,
            headers={"Content-Type": "application/json"},
            json={},
            timeout=10,
        )
        r.raise_for_status()
        if "AWSALBCORS" in r.cookies:
            headers["Cookie"] = f"AWSALBCORS={r.cookies['AWSALBCORS']}"

        token = self._token
        msg_count = 0

        def token_factory() -> str:
            return token

        def on_message(msg: list[Any] | CompletionMessage) -> None:
            nonlocal msg_count
            msg_count += 1
            self._apply_message(msg)

        conn = (
            HubConnectionBuilder()
            .with_url(
                _CONNECTION_URL,
                options={
                    "verify_ssl": True,
                    "access_token_factory": token_factory,
                    "headers": headers,
                },
            )
            .build()
        )
        connected = threading.Event()

        def on_open() -> None:
            connected.set()
            conn.send("Subscribe", [_TOPICS], on_invocation=on_message)

        conn.on_open(on_open)
        conn.on("feed", on_message)

        conn.start()
        try:
            if not connected.wait(timeout=_CONNECT_TIMEOUT_S):
                raise TimeoutError("F1 SignalR connection timed out")
            time.sleep(max(collect_s, 0.5))
        finally:
            with contextlib.suppress(Exception):
                conn.stop()

        now = time.monotonic()
        with self._lock:
            self._last_refresh = now
            self._snapshot.last_update = now
            self._snapshot.connected = msg_count > 0

    def _apply_message(self, msg: list[Any] | CompletionMessage) -> None:
        with self._lock:
            if isinstance(msg, CompletionMessage):
                result = msg.result or {}
                for topic, payload in result.items():
                    if isinstance(payload, dict):
                        self._snapshot.apply_topic(topic, payload)
                return
            if isinstance(msg, list) and len(msg) >= 2:
                topic = msg[0]
                payload = msg[1]
                if isinstance(topic, str) and isinstance(payload, dict):
                    self._snapshot.apply_topic(topic, payload)

    def _copy_snapshot(self) -> TimingSnapshot:
        with self._lock:
            return self._snapshot.copy()

    # ------------------------------------------------------------------
    # Status helpers (OpenF1-compatible)
    # ------------------------------------------------------------------

    def is_rate_limited(self) -> bool:
        return False

    def is_auth_required(self) -> bool:
        snap = self._copy_snapshot()
        return self.is_connected() and not snap.has_timing_data and not snap.drivers()

    def is_connected(self) -> bool:
        snap = self._copy_snapshot()
        if self._last_refresh <= 0:
            return False
        if snap.has_timing_data or snap.drivers() or snap.session_meta():
            return True
        return snap.connected and (time.monotonic() - self._last_refresh) < 120

    def has_data(self) -> bool:
        snap = self._copy_snapshot()
        return snap.has_timing_data or bool(snap.drivers()) or snap.session_meta() is not None

    def last_error(self) -> str | None:
        return self._error

    # ------------------------------------------------------------------
    # Session discovery
    # ------------------------------------------------------------------

    def get_latest_session(self) -> dict[str, Any] | None:
        return self._copy_snapshot().session_meta()

    def get_sessions(self, year: int) -> list[dict[str, Any]]:
        meta = self._copy_snapshot().session_meta()
        if meta is None:
            return []
        meta_year = meta.get("year")
        if meta_year and int(meta_year) != int(year):
            return []
        return [meta]

    def get_session(self, session_key: int) -> dict[str, Any] | None:
        meta = self._copy_snapshot().session_meta()
        if meta is None:
            return None
        if int(meta.get("session_key", -1)) == int(session_key):
            return meta
        return None

    # ------------------------------------------------------------------
    # Lap data
    # ------------------------------------------------------------------

    def get_laps(self, session_key: int, driver_number: int | None = None) -> list[dict[str, Any]]:
        if driver_number is None:
            return []
        lap = self._copy_snapshot().latest_lap(int(driver_number))
        return [lap] if lap else []

    def get_latest_lap(self, session_key: int, driver_number: int) -> dict[str, Any] | None:
        return self._copy_snapshot().latest_lap(int(driver_number))

    # ------------------------------------------------------------------
    # Stints
    # ------------------------------------------------------------------

    def get_stints(
        self, session_key: int, driver_number: int | None = None
    ) -> list[dict[str, Any]]:
        snap = self._copy_snapshot()
        if driver_number is not None:
            return snap.stints_for_driver(int(driver_number))
        return snap.all_stints()

    def get_current_stint(
        self, session_key: int, driver_number: int, current_lap: int
    ) -> dict[str, Any] | None:
        return self._copy_snapshot().current_stint(int(driver_number), int(current_lap))

    # ------------------------------------------------------------------
    # Position
    # ------------------------------------------------------------------

    def get_positions(
        self, session_key: int, driver_number: int | None = None
    ) -> list[dict[str, Any]]:
        if driver_number is None:
            return []
        pos = self._copy_snapshot().latest_position(int(driver_number))
        return [{"position": pos}] if pos is not None else []

    def get_latest_position(self, session_key: int, driver_number: int) -> int | None:
        return self._copy_snapshot().latest_position(int(driver_number))

    # ------------------------------------------------------------------
    # Pit stops
    # ------------------------------------------------------------------

    def get_pit_stops(
        self, session_key: int, driver_number: int | None = None
    ) -> list[dict[str, Any]]:
        if driver_number is None:
            return []
        count = self._copy_snapshot().pit_stop_count(int(driver_number))
        return [{"lap_number": i + 1} for i in range(count)]

    # ------------------------------------------------------------------
    # Race control
    # ------------------------------------------------------------------

    def get_race_control(self, session_key: int) -> list[dict[str, Any]]:
        return self._copy_snapshot().race_control_messages

    def is_safety_car_active(self, session_key: int, current_lap: int) -> bool:
        return self._copy_snapshot().is_safety_car_active()

    # ------------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------------

    def get_weather(self, session_key: int) -> list[dict[str, Any]]:
        w = self._copy_snapshot().latest_weather()
        return [w] if w else []

    def get_latest_weather(self, session_key: int) -> dict[str, Any] | None:
        return self._copy_snapshot().latest_weather()

    # ------------------------------------------------------------------
    # Drivers
    # ------------------------------------------------------------------

    def get_drivers(self, session_key: int) -> list[dict[str, Any]]:
        return self._copy_snapshot().drivers()
