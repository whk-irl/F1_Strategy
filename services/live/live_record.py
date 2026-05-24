"""Persistent live-race logs in OpenF1-compatible layout on S3.

Canonical module for live S3 logging.  The Streamlit app imports this file
directly (``live_record``) to avoid Streamlit Cloud caching stale copies of
older modules such as ``race_log``.

Each OpenF1-style *endpoint* is stored as one Parquet file per session:

    pitwall_live/laps/session_key=9839/data.parquet
    pitwall_live/position/session_key=9839/data.parquet
    pitwall_live/stints/session_key=9839/data.parquet
    pitwall_live/weather/session_key=9839/data.parquet
    pitwall_live/strategy_predictions/session_key=9839/data.parquet
    pitwall_live/sessions/session_key=9839/data.parquet

Column names mirror the OpenF1 REST API where applicable (``session_key``,
``meeting_key``, ``driver_number``, ``date``, ``lap_number``, ΓÇª) so logs can
be joined with OpenF1 exports or queried like API responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from services.ingestion.config import IngestionSettings
from services.ingestion.storage import ObjectStorage
from services.live.obs_builder import DriverLiveState, update_from_openf1

__all__ = [
    "ENDPOINTS",
    "LOG_PREFIX",
    "LiveTickContext",
    "append_live_tick",
    "endpoint_key",
    "get_storage",
    "list_races",
    "list_sessions",
    "load_endpoint",
    "load_race",
    "load_session_predictions",
    "reset_storage",
]

LOG_PREFIX = "pitwall_live"

LAPS_COLS: list[str] = [
    "date",
    "session_key",
    "meeting_key",
    "driver_number",
    "lap_number",
    "lap_duration",
    "is_pit_out_lap",
]

POSITION_COLS: list[str] = [
    "date",
    "session_key",
    "meeting_key",
    "driver_number",
    "position",
    "lap_number",
]

STINTS_COLS: list[str] = [
    "date",
    "session_key",
    "meeting_key",
    "driver_number",
    "stint_number",
    "compound",
    "lap_start",
    "lap_end",
    "tyre_age_at_start",
]

WEATHER_COLS: list[str] = [
    "date",
    "session_key",
    "meeting_key",
    "track_temperature",
    "rainfall",
]

STRATEGY_COLS: list[str] = [
    "date",
    "session_key",
    "meeting_key",
    "driver_number",
    "lap_number",
    "position",
    "compound",
    "tyre_life",
    "pit_stops",
    "sc_active",
    "wet_fraction",
    "recommended_action",
    "recommended_label",
    "prob_stay",
    "prob_soft",
    "prob_medium",
    "prob_hard",
    "model_key",
    "model_type",
] + [f"obs_{i:02d}" for i in range(21)]

SESSIONS_COLS: list[str] = [
    "date",
    "session_key",
    "meeting_key",
    "session_name",
    "session_type",
    "year",
    "country_name",
    "circuit_short_name",
    "meeting_name",
    "date_start",
    "date_end",
    "total_laps",
]

ENDPOINTS: dict[str, list[str]] = {
    "laps": LAPS_COLS,
    "position": POSITION_COLS,
    "stints": STINTS_COLS,
    "weather": WEATHER_COLS,
    "strategy_predictions": STRATEGY_COLS,
    "sessions": SESSIONS_COLS,
}

_storage: ObjectStorage | None = None


def get_storage() -> ObjectStorage:
    """Return a configured ObjectStorage instance (lazy singleton)."""
    global _storage
    if _storage is None:
        settings = IngestionSettings()
        _storage = ObjectStorage(settings)
    return _storage


def reset_storage() -> None:
    """Clear the cached storage client (used by tests)."""
    global _storage
    _storage = None


def endpoint_key(endpoint: str, session_key: int) -> str:
    """Return the S3 object key for an OpenF1-style endpoint file."""
    if endpoint not in ENDPOINTS:
        raise ValueError(f"Unknown endpoint: {endpoint}")
    return f"{LOG_PREFIX}/{endpoint}/session_key={int(session_key)}/data.parquet"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _append_rows(
    endpoint: str,
    session_key: int,
    rows: list[dict[str, Any]],
    *,
    dedupe_on: list[str] | None = None,
) -> str | None:
    """Append rows to an endpoint parquet; return the S3 key or None if empty."""
    if not rows:
        return None
    columns = ENDPOINTS[endpoint]
    storage = get_storage()
    key = endpoint_key(endpoint, session_key)
    new_df = pd.DataFrame(rows, columns=columns)

    try:
        existing = storage.read_parquet(key)
        for col in columns:
            if col not in existing.columns:
                existing[col] = pd.NA
        combined = pd.concat([existing[columns], new_df], ignore_index=True)
    except KeyError:
        combined = new_df

    if dedupe_on:
        combined = combined.drop_duplicates(subset=dedupe_on, keep="last")

    storage.write_parquet(combined, key)
    return key


@dataclass
class LiveTickContext:
    """Inputs for one full-field logging tick during a live race."""

    session_meta: dict[str, Any]
    client: Any
    drivers: list[dict[str, Any]]
    policy: Any
    model_type: str
    model_key: str
    sc_active: bool
    field_stints: list[dict[str, Any]]
    weather: dict[str, Any] | None
    pit_loss_s: float
    recommend_fn: Any
    driver_states: dict[int, DriverLiveState]


def append_live_tick(ctx: LiveTickContext) -> list[str]:
    """Log a full-field snapshot across all OpenF1-style endpoints.

    Writes laps, position, stints, weather, strategy predictions, and session
    metadata for every driver on the timing feed.

    Returns:
        List of S3 keys written this tick.
    """
    session_key = int(ctx.session_meta["session_key"])
    meeting_key = ctx.session_meta.get("meeting_key")
    meeting_key_val = int(meeting_key) if meeting_key is not None else None
    now = _utc_now_iso()
    total_laps = int(ctx.session_meta.get("total_laps") or 0)
    n_drivers = len(ctx.drivers)

    lap_rows: list[dict[str, Any]] = []
    pos_rows: list[dict[str, Any]] = []
    stint_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    for driver in ctx.drivers:
        driver_number = int(driver["driver_number"])
        state = ctx.driver_states.get(driver_number)
        if state is None:
            state = DriverLiveState(
                driver_number=driver_number,
                total_laps=max(total_laps, 1),
                pit_loss_s=ctx.pit_loss_s,
                n_drivers=n_drivers,
            )
            ctx.driver_states[driver_number] = state

        try:
            lap_record = ctx.client.get_latest_lap(session_key, driver_number)
            stint_record = ctx.client.get_current_stint(
                session_key, driver_number, state.current_lap
            )
            position = ctx.client.get_latest_position(session_key, driver_number)
            pit_stops = len(ctx.client.get_pit_stops(session_key, driver_number))
        except Exception:  # noqa: BLE001
            continue

        update_from_openf1(
            state,
            lap_record,
            stint_record,
            position,
            pit_stops,
            ctx.sc_active,
            ctx.field_stints,
            ctx.weather,
        )

        if state.current_lap <= 0:
            continue

        lap_duration = None
        if lap_record is not None:
            lap_duration = lap_record.get("lap_duration")

        lap_rows.append(
            {
                "date": now,
                "session_key": session_key,
                "meeting_key": meeting_key_val,
                "driver_number": driver_number,
                "lap_number": state.current_lap,
                "lap_duration": float(lap_duration) if lap_duration is not None else None,
                "is_pit_out_lap": False,
            }
        )

        if position is not None:
            pos_rows.append(
                {
                    "date": now,
                    "session_key": session_key,
                    "meeting_key": meeting_key_val,
                    "driver_number": driver_number,
                    "position": int(position),
                    "lap_number": state.current_lap,
                }
            )

        if stint_record is not None:
            compound = stint_record.get("compound")
            lap_start = int(stint_record.get("lap_start") or 1)
            lap_end_raw = stint_record.get("lap_end")
            stint_rows.append(
                {
                    "date": now,
                    "session_key": session_key,
                    "meeting_key": meeting_key_val,
                    "driver_number": driver_number,
                    "stint_number": max(state.pit_stops + 1, 1),
                    "compound": compound,
                    "lap_start": lap_start,
                    "lap_end": int(lap_end_raw) if lap_end_raw is not None else None,
                    "tyre_age_at_start": max(state.current_lap - lap_start, 0),
                }
            )

        obs = state.build_obs()
        action, label, probs, qv = ctx.recommend_fn(obs, ctx.policy, ctx.model_type)
        compound_name = {0: "SOFT", 1: "MEDIUM", 2: "HARD", 3: "INTER", 4: "WET", 5: "?"}.get(
            state.compound_encoded, "?"
        )
        row: dict[str, Any] = {
            "date": now,
            "session_key": session_key,
            "meeting_key": meeting_key_val,
            "driver_number": driver_number,
            "lap_number": state.current_lap,
            "position": state.position,
            "compound": compound_name,
            "tyre_life": state.tyre_life,
            "pit_stops": state.pit_stops,
            "sc_active": ctx.sc_active,
            "wet_fraction": state.wet_fraction,
            "recommended_action": int(action),
            "recommended_label": str(label),
            "prob_stay": float(probs.get("Stay out", 0.0)),
            "prob_soft": float(probs.get("Pit ΓÇö SOFT", 0.0)),
            "prob_medium": float(probs.get("Pit ΓÇö MEDIUM", 0.0)),
            "prob_hard": float(probs.get("Pit ΓÇö HARD", 0.0)),
            "model_key": ctx.model_key,
            "model_type": ctx.model_type,
        }
        obs_arr = np.asarray(obs, dtype=float).flatten()
        for i in range(21):
            row[f"obs_{i:02d}"] = float(obs_arr[i]) if i < obs_arr.size else float("nan")
        pred_rows.append(row)

    written: list[str] = []
    for endpoint, rows, dedupe in (
        ("laps", lap_rows, ["driver_number", "lap_number"]),
        ("position", pos_rows, ["driver_number", "lap_number"]),
        ("stints", stint_rows, ["driver_number", "stint_number", "lap_start"]),
        ("strategy_predictions", pred_rows, ["driver_number", "lap_number"]),
    ):
        key = _append_rows(endpoint, session_key, rows, dedupe_on=dedupe)
        if key:
            written.append(key)

    if ctx.weather is not None:
        wrow = {
            "date": now,
            "session_key": session_key,
            "meeting_key": meeting_key_val,
            "track_temperature": ctx.weather.get("track_temperature"),
            "rainfall": ctx.weather.get("rainfall"),
        }
        key = _append_rows("weather", session_key, [wrow], dedupe_on=None)
        if key:
            written.append(key)

    session_row = {
        "date": now,
        "session_key": session_key,
        "meeting_key": meeting_key_val,
        "session_name": ctx.session_meta.get("session_name"),
        "session_type": ctx.session_meta.get("session_type"),
        "year": ctx.session_meta.get("year"),
        "country_name": ctx.session_meta.get("country_name"),
        "circuit_short_name": ctx.session_meta.get("circuit_short_name"),
        "meeting_name": ctx.session_meta.get("meeting_name"),
        "date_start": ctx.session_meta.get("date_start"),
        "date_end": ctx.session_meta.get("date_end"),
        "total_laps": total_laps,
    }
    key = _append_rows("sessions", session_key, [session_row], dedupe_on=["session_key"])
    if key:
        written.append(key)

    return written


def list_sessions() -> list[dict[str, Any]]:
    """List sessions that have strategy prediction logs."""
    storage = get_storage()
    by_session: dict[int, dict[str, Any]] = {}
    prefix = f"{LOG_PREFIX}/strategy_predictions/"
    for key in storage.list_keys(prefix):
        parts = key.split("/")
        if len(parts) < 3:
            continue
        session_part = parts[2]
        if not session_part.startswith("session_key="):
            continue
        try:
            session_key = int(session_part.split("=", 1)[1])
        except (ValueError, IndexError):
            continue
        by_session.setdefault(session_key, {"session_key": session_key, "keys": []})
        by_session[session_key]["keys"].append(key)

    sessions_meta = load_all_sessions_meta()
    for session_key, entry in by_session.items():
        meta = sessions_meta.get(session_key, {})
        entry.update(meta)
        try:
            preds = load_endpoint("strategy_predictions", session_key)
            entry["n_rows"] = len(preds)
            entry["n_drivers"] = int(preds["driver_number"].nunique()) if not preds.empty else 0
        except Exception:  # noqa: BLE001
            entry["n_rows"] = 0
            entry["n_drivers"] = 0

    return sorted(
        by_session.values(),
        key=lambda s: (s.get("year") or 0, s.get("session_key") or 0),
        reverse=True,
    )


def load_all_sessions_meta() -> dict[int, dict[str, Any]]:
    """Load the latest metadata row per session_key from sessions endpoint files."""
    storage = get_storage()
    meta: dict[int, dict[str, Any]] = {}
    for key in storage.list_keys(f"{LOG_PREFIX}/sessions/"):
        try:
            df = storage.read_parquet(key)
        except Exception:  # noqa: BLE001
            continue
        if df.empty or "session_key" not in df.columns:
            continue
        row = df.sort_values("date").iloc[-1]
        sk = int(row["session_key"])
        meta[sk] = row.to_dict()
    return meta


def load_endpoint(endpoint: str, session_key: int) -> pd.DataFrame:
    """Load one endpoint parquet for a session."""
    storage = get_storage()
    key = endpoint_key(endpoint, session_key)
    try:
        return storage.read_parquet(key)
    except KeyError:
        return pd.DataFrame(columns=ENDPOINTS[endpoint])


def load_session_predictions(session_key: int) -> pd.DataFrame:
    """Load strategy predictions for a session (primary log for the UI)."""
    return load_endpoint("strategy_predictions", session_key)


def list_races() -> list[dict[str, Any]]:
    """Backward-compatible alias for :func:`list_sessions`."""
    races: list[dict[str, Any]] = []
    for s in list_sessions():
        label = s.get("meeting_name") or s.get("country_name") or f"Session {s['session_key']}"
        session_name = s.get("session_name") or "Race"
        races.append(
            {
                "year": int(s.get("year") or datetime.now(timezone.utc).year),
                "race_folder": f"{label}_{session_name}".replace(" ", "_"),
                "session_key": s["session_key"],
                "n_drivers": s.get("n_drivers", 0),
                "keys": s.get("keys", []),
            }
        )
    return races


def load_race(year: int, race_folder: str) -> pd.DataFrame:
    """Backward-compatible loader ΓÇö maps legacy race_folder to session_key via metadata."""
    for session in list_sessions():
        label = session.get("meeting_name") or session.get("country_name") or ""
        session_name = session.get("session_name") or "Race"
        folder = f"{label}_{session_name}".replace(" ", "_")
        if int(session.get("year") or 0) == int(year) and folder == race_folder:
            return _predictions_for_ui(load_session_predictions(int(session["session_key"])))
    return pd.DataFrame()


def _predictions_for_ui(df: pd.DataFrame) -> pd.DataFrame:
    """Map OpenF1-style strategy columns to legacy Prediction Log UI names."""
    if df.empty:
        return df
    out = df.copy()
    out["timestamp_utc"] = out["date"]
    out["lap"] = out["lap_number"]
    out["compound_name"] = out.get("compound", pd.Series(dtype=str))
    out["driver_label"] = out["driver_number"].apply(lambda n: f"#{n}")
    out["gp_name"] = out.get("meeting_name", pd.Series(dtype=str))
    out["circuit"] = out.get("circuit_short_name", pd.Series(dtype=str))
    return out
