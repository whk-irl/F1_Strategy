"""Persistent log of live-race strategy predictions.

Each call to :func:`append_prediction` writes one row capturing the model's
recommendation for a single lap.  Logs are stored as Parquet objects in S3
under the ``live_logs/`` prefix of ``PITWALL_AWS_S3_BUCKET`` (or the MinIO
bucket when ``PITWALL_STORAGE_BACKEND=minio``) so a portfolio retrospective
can replay exactly what the model called in real time.

Append-on-lap-change is enforced by the caller (Streamlit tab) — this module
just persists whatever it's handed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from services.ingestion.config import IngestionSettings
from services.ingestion.storage import ObjectStorage
from services.live.obs_builder import DriverLiveState

# Explicit __all__ forces Streamlit Cloud to invalidate its module file cache
# when these symbols change.  Without it, freshly-added names like LOG_PREFIX
# may not be visible until the app is fully redeployed (see commit 817b795
# for the same workaround applied to the sequence-tire prediction module).
__all__ = [
    "LOG_COLS",
    "LOG_PREFIX",
    "PredictionRow",
    "append_prediction",
    "get_storage",
    "list_logs",
    "load_log",
    "log_key",
    "reset_storage",
]

# S3 key prefix for live prediction logs.
LOG_PREFIX = "live_logs"

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


# Module-level singleton — created lazily so import never fails when S3 is
# unconfigured.  Streamlit caches this at the call site via @st.cache_resource.
_storage: ObjectStorage | None = None


def get_storage() -> ObjectStorage:
    """Return a configured ObjectStorage instance.  Constructed on first use.

    Raises:
        RuntimeError: If the storage backend can't be initialised (missing
            bucket, bad credentials, network).  Callers should surface this
            to the UI so the user knows logging is disabled for the session.
    """
    global _storage
    if _storage is None:
        settings = IngestionSettings()
        _storage = ObjectStorage(settings)
    return _storage


def reset_storage() -> None:
    """Clear the cached storage client (used by tests)."""
    global _storage
    _storage = None


def log_key(session_key: int, driver_number: int) -> str:
    """Return the S3 object key for a (session, driver) log file."""
    return f"{LOG_PREFIX}/{int(session_key)}_{int(driver_number)}.parquet"


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


def append_prediction(row: PredictionRow) -> str:
    """Append one prediction row to the S3 log object.

    Read-modify-write: parquet doesn't support cheap appends, so we pull the
    existing object (if any), concatenate, and re-upload.  Fine for
    once-per-lap cadence.  Returns the S3 key.
    """
    storage = get_storage()
    key = log_key(row.session_key, row.driver_number)
    new_df = pd.DataFrame([row.to_dict()], columns=LOG_COLS)

    try:
        existing = storage.read_parquet(key)
        for col in LOG_COLS:
            if col not in existing.columns:
                existing[col] = pd.NA
        combined = pd.concat([existing[LOG_COLS], new_df], ignore_index=True)
    except KeyError:
        combined = new_df

    storage.write_parquet(combined, key)
    return key


def load_log(session_key: int, driver_number: int) -> pd.DataFrame:
    """Load a single (session, driver) log from S3.  Empty DataFrame if missing."""
    storage = get_storage()
    key = log_key(session_key, driver_number)
    try:
        return storage.read_parquet(key)
    except KeyError:
        return pd.DataFrame(columns=LOG_COLS)


def list_logs() -> list[dict[str, Any]]:
    """Return a summary of all log objects in S3 under :data:`LOG_PREFIX`.

    Each entry has: session_key, driver_number, rows, first_lap, last_lap,
    last_updated, gp_name, session_name, driver_label, key.  Sorted newest first.
    """
    storage = get_storage()
    out: list[dict[str, Any]] = []
    for key in storage.list_keys(f"{LOG_PREFIX}/"):
        try:
            df = storage.read_parquet(key)
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
                "key": key,
            }
        )
    out.sort(key=lambda r: r["last_updated"], reverse=True)
    return out
