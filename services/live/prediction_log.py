"""Persistent log of live-race strategy predictions.

Each call to :func:`append_prediction` writes one row capturing the model's
recommendation for a single lap.  Logs are stored as Parquet files under
``data/live_logs/{session_key}_{driver_number}.parquet`` so a portfolio
retrospective can replay exactly what the model called in real time.

Append-on-lap-change is enforced by the caller (Streamlit tab) — this module
just persists whatever it's handed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from services.live.obs_builder import DriverLiveState

# Columns persisted to parquet, in stable order.
_BASE_COLS: list[str] = [
    "timestamp_utc",
    "session_key",
    "session_name",
    "gp_name",
    "circuit",
    "driver_number",
    "driver_label",
    "model_key",
    "model_type",
    "lap",
    "total_laps",
    "position",
    "compound_encoded",
    "compound_name",
    "tyre_life",
    "pit_stops",
    "sc_active",
    "sc_probability",
    "wet_fraction",
    "gap_ahead_s",
    "gap_behind_s",
    "recommended_action",
    "recommended_label",
    "prob_stay",
    "prob_soft",
    "prob_medium",
    "prob_hard",
    "q_stay",
    "q_soft",
    "q_medium",
    "q_hard",
]
_OBS_COLS: list[str] = [f"obs_{i:02d}" for i in range(21)]
LOG_COLS: list[str] = _BASE_COLS + _OBS_COLS


def _logs_dir() -> Path:
    """Return the live-logs directory, honouring ``PITWALL_LIVE_LOG_DIR`` env."""
    override = os.getenv("PITWALL_LIVE_LOG_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "data" / "live_logs"


def log_path(session_key: int, driver_number: int) -> Path:
    """Return the parquet path for a (session, driver) log file."""
    return _logs_dir() / f"{int(session_key)}_{int(driver_number)}.parquet"


@dataclass
class PredictionRow:
    """One captured prediction.  Mirrors :data:`LOG_COLS`."""

    session_key: int
    session_name: str
    gp_name: str
    circuit: str
    driver_number: int
    driver_label: str
    model_key: str
    model_type: str
    state: DriverLiveState
    obs: np.ndarray
    recommended_action: int
    recommended_label: str
    probs: dict[str, float]
    q_values: dict[str, float] | None

    def to_dict(self) -> dict[str, Any]:
        """Flatten this row to the persisted schema."""
        compound_name = {0: "SOFT", 1: "MEDIUM", 2: "HARD", 3: "INTER", 4: "WET", 5: "?"}[
            self.state.compound_encoded
        ]
        row: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session_key": int(self.session_key),
            "session_name": self.session_name,
            "gp_name": self.gp_name,
            "circuit": self.circuit,
            "driver_number": int(self.driver_number),
            "driver_label": self.driver_label,
            "model_key": self.model_key,
            "model_type": self.model_type,
            "lap": int(self.state.current_lap),
            "total_laps": int(self.state.total_laps),
            "position": int(self.state.position),
            "compound_encoded": int(self.state.compound_encoded),
            "compound_name": compound_name,
            "tyre_life": int(self.state.tyre_life),
            "pit_stops": int(self.state.pit_stops),
            "sc_active": bool(self.state.sc_active),
            "sc_probability": float(self.state.sc_probability),
            "wet_fraction": float(self.state.wet_fraction),
            "gap_ahead_s": float(self.state.gap_ahead_s),
            "gap_behind_s": float(self.state.gap_behind_s),
            "recommended_action": int(self.recommended_action),
            "recommended_label": self.recommended_label,
            "prob_stay": float(self.probs.get("Stay out", 0.0)),
            "prob_soft": float(self.probs.get("Pit — SOFT", 0.0)),
            "prob_medium": float(self.probs.get("Pit — MEDIUM", 0.0)),
            "prob_hard": float(self.probs.get("Pit — HARD", 0.0)),
            "q_stay": float(self.q_values["Stay out"]) if self.q_values else float("nan"),
            "q_soft": float(self.q_values["Pit — SOFT"]) if self.q_values else float("nan"),
            "q_medium": float(self.q_values["Pit — MEDIUM"]) if self.q_values else float("nan"),
            "q_hard": float(self.q_values["Pit — HARD"]) if self.q_values else float("nan"),
        }
        obs_arr = np.asarray(self.obs, dtype=float).flatten()
        for i, col in enumerate(_OBS_COLS):
            row[col] = float(obs_arr[i]) if i < obs_arr.size else float("nan")
        return row


def append_prediction(row: PredictionRow) -> Path:
    """Append one prediction row to the appropriate parquet file.

    Re-reads the existing file (if any), concatenates, and re-writes — fine for
    the once-per-lap cadence we run at.  Returns the file path.
    """
    path = log_path(row.session_key, row.driver_number)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame([row.to_dict()], columns=LOG_COLS)

    if path.exists():
        existing = pd.read_parquet(path)
        for col in LOG_COLS:
            if col not in existing.columns:
                existing[col] = pd.NA
        combined = pd.concat([existing[LOG_COLS], new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_parquet(path, index=False)
    return path


def load_log(session_key: int, driver_number: int) -> pd.DataFrame:
    """Load a single (session, driver) log.  Returns empty DataFrame if missing."""
    path = log_path(session_key, driver_number)
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLS)
    return pd.read_parquet(path)


def list_logs() -> list[dict[str, Any]]:
    """Return a summary of all logs on disk.

    Each entry has: session_key, driver_number, rows, first_lap, last_lap,
    last_updated, gp_name, session_name, path.  Sorted newest first.
    """
    out: list[dict[str, Any]] = []
    base = _logs_dir()
    if not base.exists():
        return out
    for path in base.glob("*.parquet"):
        try:
            df = pd.read_parquet(path)
        except Exception:  # noqa: BLE001 — skip corrupt files rather than crash the tab
            continue
        if df.empty:
            continue
        last = df.iloc[-1]
        out.append(
            {
                "session_key": int(last.get("session_key", 0)),
                "driver_number": int(last.get("driver_number", 0)),
                "driver_label": str(last.get("driver_label", "")),
                "gp_name": str(last.get("gp_name", "")),
                "session_name": str(last.get("session_name", "")),
                "rows": int(len(df)),
                "first_lap": int(df["lap"].min()) if "lap" in df.columns else 0,
                "last_lap": int(df["lap"].max()) if "lap" in df.columns else 0,
                "last_updated": str(last.get("timestamp_utc", "")),
                "path": str(path),
            }
        )
    out.sort(key=lambda r: r["last_updated"], reverse=True)
    return out
