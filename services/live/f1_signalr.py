"""F1 official live timing client via SignalR.

Connects to the same feed that powers
https://www.formula1.com/en/timing/f1-live and exposes an OpenF1-compatible
interface for the Streamlit live tab.

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
_RECONNECT_DELAY_S = 5.0
_CONNECT_TIMEOUT_S = 15.0

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
    """Stateful client for the official F1 live timing SignalR feed.

    A background thread maintains the WebSocket connection and merges delta
    updates into a :class:`TimingSnapshot`.  Public methods mirror
    :class:`services.live.openf1_api.OpenF1Client` so the frontend can swap
    providers without changing observation logic.

    Args:
        subscription_token: Optional F1TV subscription JWT.  When empty the
            client uses the same free feed as
            https://www.formula1.com/en/timing/f1-live (positions, lap times,
            compounds, weather).  A token is only needed for F1TV-only extras.
    """

    def __init__(self, subscription_token: str | None = None) -> None:
        self._token = resolve_subscription_token(subscription_token)
        self._snapshot = TimingSnapshot()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._connection: Any = None
        self._error: str | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def ensure_started(self) -> None:
        """Start the background SignalR thread if not already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="f1-signalr", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread and close the WebSocket."""
        self._stop.set()
        if self._connection is not None:
            with contextlib.suppress(Exception):
                self._connection.stop()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._connect_once()
            except Exception as exc:  # noqa: BLE001
                self._error = str(exc)
                logger.warning("F1 SignalR connection error: %s", exc)
                with self._lock:
                    self._snapshot.connected = False
            if not self._stop.is_set():
                time.sleep(_RECONNECT_DELAY_S)

    def _connect_once(self) -> None:
        headers: dict[str, str] = {}
        # POST negotiate (OPTIONS returns 405 on current F1 infra).  Grabs the
        # AWSALBCORS sticky-session cookie that signalrcore also needs.
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

        def token_factory() -> str:
            return token

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
        self._connection = conn
        connected = threading.Event()

        def on_open() -> None:
            with self._lock:
                self._snapshot.connected = True
                self._error = None
            connected.set()
            conn.send("Subscribe", [_TOPICS], on_invocation=self._on_message)

        def on_close() -> None:
            connected.clear()
            self._mark_disconnected()

        conn.on_open(on_open)
        conn.on("feed", self._on_message)
        conn.on_close(on_close)

        conn.start()
        if not connected.wait(timeout=_CONNECT_TIMEOUT_S):
            conn.stop()
            raise TimeoutError("F1 SignalR connection timed out")

        while not self._stop.is_set() and connected.is_set():
            time.sleep(0.5)

        with contextlib.suppress(Exception):
            conn.stop()
        self._mark_disconnected()

    def _mark_disconnected(self) -> None:
        with self._lock:
            self._snapshot.connected = False

    def _on_message(self, msg: list[Any] | CompletionMessage) -> None:
        now = time.monotonic()
        with self._lock:
            self._snapshot.last_update = now
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
        """True when connected but no timing lines have arrived yet."""
        snap = self._copy_snapshot()
        return snap.connected and not snap.has_timing_data and not snap.drivers()

    def is_connected(self) -> bool:
        return self._copy_snapshot().connected

    def has_data(self) -> bool:
        snap = self._copy_snapshot()
        return snap.has_timing_data or bool(snap.drivers()) or snap.session_meta() is not None

    def last_error(self) -> str | None:
        return self._error

    # ------------------------------------------------------------------
    # Session discovery
    # ------------------------------------------------------------------

    def get_latest_session(self) -> dict[str, Any] | None:
        self.ensure_started()
        return self._copy_snapshot().session_meta()

    def get_sessions(self, year: int) -> list[dict[str, Any]]:
        self.ensure_started()
        meta = self._copy_snapshot().session_meta()
        if meta is None:
            return []
        meta_year = meta.get("year")
        if meta_year and int(meta_year) != int(year):
            return []
        return [meta]

    def get_session(self, session_key: int) -> dict[str, Any] | None:
        self.ensure_started()
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
        self.ensure_started()
        if driver_number is None:
            return []
        lap = self._copy_snapshot().latest_lap(int(driver_number))
        return [lap] if lap else []

    def get_latest_lap(self, session_key: int, driver_number: int) -> dict[str, Any] | None:
        self.ensure_started()
        return self._copy_snapshot().latest_lap(int(driver_number))

    # ------------------------------------------------------------------
    # Stints
    # ------------------------------------------------------------------

    def get_stints(
        self, session_key: int, driver_number: int | None = None
    ) -> list[dict[str, Any]]:
        self.ensure_started()
        snap = self._copy_snapshot()
        if driver_number is not None:
            return snap.stints_for_driver(int(driver_number))
        return snap.all_stints()

    def get_current_stint(
        self, session_key: int, driver_number: int, current_lap: int
    ) -> dict[str, Any] | None:
        self.ensure_started()
        return self._copy_snapshot().current_stint(int(driver_number), int(current_lap))

    # ------------------------------------------------------------------
    # Position
    # ------------------------------------------------------------------

    def get_positions(
        self, session_key: int, driver_number: int | None = None
    ) -> list[dict[str, Any]]:
        self.ensure_started()
        if driver_number is None:
            return []
        pos = self._copy_snapshot().latest_position(int(driver_number))
        return [{"position": pos}] if pos is not None else []

    def get_latest_position(self, session_key: int, driver_number: int) -> int | None:
        self.ensure_started()
        return self._copy_snapshot().latest_position(int(driver_number))

    # ------------------------------------------------------------------
    # Pit stops
    # ------------------------------------------------------------------

    def get_pit_stops(
        self, session_key: int, driver_number: int | None = None
    ) -> list[dict[str, Any]]:
        self.ensure_started()
        if driver_number is None:
            return []
        count = self._copy_snapshot().pit_stop_count(int(driver_number))
        return [{"lap_number": i + 1} for i in range(count)]

    # ------------------------------------------------------------------
    # Race control
    # ------------------------------------------------------------------

    def get_race_control(self, session_key: int) -> list[dict[str, Any]]:
        self.ensure_started()
        return self._copy_snapshot().race_control_messages

    def is_safety_car_active(self, session_key: int, current_lap: int) -> bool:
        self.ensure_started()
        return self._copy_snapshot().is_safety_car_active()

    # ------------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------------

    def get_weather(self, session_key: int) -> list[dict[str, Any]]:
        self.ensure_started()
        w = self._copy_snapshot().latest_weather()
        return [w] if w else []

    def get_latest_weather(self, session_key: int) -> dict[str, Any] | None:
        self.ensure_started()
        return self._copy_snapshot().latest_weather()

    # ------------------------------------------------------------------
    # Drivers
    # ------------------------------------------------------------------

    def get_drivers(self, session_key: int) -> list[dict[str, Any]]:
        self.ensure_started()
        return self._copy_snapshot().drivers()
