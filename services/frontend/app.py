"""Pitwall AI — Streamlit demo app.

Two tabs:
  • Race Replay — 2024 recorded race replayed through the PPO policy.
  • Live Race   — live lap-by-lap recommendations via the OpenF1 public API.

Run with:
    streamlit run services/frontend/app.py
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import mlflow.pyfunc
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import ml.models.strategy_policy.predict as _ppo_predict
import ml.models.strategy_policy.predict_dqn as _dqn_predict
from ml.models._loader import load_gold_seasons
from ml.models.strategy_policy.per_buffer import PrioritizedReplayBuffer
from ml.models.strategy_policy.train_dqn import DQNPER
from ml.models.tire_degradation.predict import predict_stint_degradation as _lgbm_stint_predict

# Sequence tire models are optional — if the import fails (e.g. torch build mismatch
# on the deployment platform) the app degrades to LightGBM-only mode instead of
# crashing entirely.  The actual error is logged below so it shows up in cloud logs.
_SEQ_MODELS_AVAILABLE = False
_seq_import_error: str = ""
try:
    from ml.models.tire_degradation.wrapper import SequenceTireWrapper
    from ml.models.tire_degradation.predict_sequence import (
        load_sequence_model as _load_seq_tire_model,
        predict_stint_from_history as _seq_stint_forecast,
    )
    from ml.models.tire_degradation.sequence_dataset import FEATURE_COLS as _SEQ_FEATURE_COLS
    _SEQ_MODELS_AVAILABLE = True
except Exception as _e:  # noqa: BLE001
    import traceback as _tb
    _seq_import_error = _tb.format_exc()
    SequenceTireWrapper = None  # type: ignore[assignment, misc]
    _load_seq_tire_model = None  # type: ignore[assignment]
    _seq_stint_forecast = None  # type: ignore[assignment]
    _SEQ_FEATURE_COLS: list[str] = []
from stable_baselines3 import PPO

from services.live.obs_builder import DriverLiveState, update_from_openf1
from services.live.openf1_api import OpenF1Client
from services.live.race_log import (
    LOG_PREFIX,
    PredictionRow,
    append_prediction,
    get_storage,
    list_races,
    load_race,
    race_folder_name,
    race_year,
)
from services.simulator.env import F1RaceEnv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUND_NAMES_BY_YEAR: dict[int, dict[int, str]] = {
    2022: {
        1: "Bahrain GP",
        2: "Saudi Arabian GP",
        3: "Australian GP",
        4: "Emilia Romagna GP",
        5: "Miami GP",
        6: "Spanish GP",
        7: "Monaco GP",
        8: "Azerbaijan GP",
        9: "Canadian GP",
        10: "British GP",
        11: "Austrian GP",
        12: "French GP",
        13: "Hungarian GP",
        14: "Belgian GP",
        15: "Dutch GP",
        16: "Italian GP",
        17: "Singapore GP",
        18: "Japanese GP",
        19: "United States GP",
        20: "Mexico City GP",
        21: "São Paulo GP",
        22: "Abu Dhabi GP",
    },
    2023: {
        1: "Bahrain GP",
        2: "Saudi Arabian GP",
        3: "Australian GP",
        4: "Azerbaijan GP",
        5: "Miami GP",
        6: "Monaco GP",
        7: "Spanish GP",
        8: "Canadian GP",
        9: "Austrian GP",
        10: "British GP",
        11: "Hungarian GP",
        12: "Belgian GP",
        13: "Dutch GP",
        14: "Italian GP",
        15: "Singapore GP",
        16: "Japanese GP",
        17: "Qatar GP",
        18: "United States GP",
        19: "Mexico City GP",
        20: "São Paulo GP",
        21: "Las Vegas GP",
        22: "Abu Dhabi GP",
    },
    2024: {
        1: "Bahrain GP",
        2: "Saudi Arabian GP",
        3: "Australian GP",
        4: "Japanese GP",
        5: "Chinese GP",
        6: "Miami GP",
        7: "Emilia Romagna GP",
        8: "Monaco GP",
        9: "Canadian GP",
        10: "Spanish GP",
        11: "Austrian GP",
        12: "British GP",
        13: "Hungarian GP",
        14: "Belgian GP",
        15: "Dutch GP",
        16: "Italian GP",
        17: "Azerbaijan GP",
        18: "Singapore GP",
        19: "United States GP",
        20: "Mexico City GP",
        21: "São Paulo GP",
        22: "Las Vegas GP",
        23: "Qatar GP",
        24: "Abu Dhabi GP",
    },
    2025: {
        1: "Australian GP",
        2: "Chinese GP",
        3: "Japanese GP",
        4: "Bahrain GP",
        5: "Saudi Arabian GP",
        6: "Miami GP",
        7: "Emilia Romagna GP",
        8: "Monaco GP",
        9: "Spanish GP",
        10: "Canadian GP",
        11: "Austrian GP",
        12: "British GP",
        13: "Belgian GP",
        14: "Hungarian GP",
        15: "Dutch GP",
        16: "Italian GP",
        17: "Azerbaijan GP",
        18: "Singapore GP",
        19: "United States GP",
        20: "Mexico City GP",
        21: "São Paulo GP",
        22: "Las Vegas GP",
        23: "Qatar GP",
        24: "Abu Dhabi GP",
    },
}

_TIRE_FORECAST_OPTIONS: dict[str, str] = {
    "LightGBM (baseline)": "lgbm",
    "TCN+GRU (sequence)": "tcn_gru",
    "PatchTST (transformer)": "patch_tst",
}
_TIRE_FORECAST_COLORS: dict[str, str] = {
    "lgbm": "#44aaff",
    "tcn_gru": "#ff6644",
    "patch_tst": "#44ff88",
}

COMPOUND_LABEL: dict[int, str] = {
    0: "SOFT",
    1: "MEDIUM",
    2: "HARD",
    3: "INTER",
    4: "WET",
    5: "UNKNOWN",
}
COMPOUND_COLOR: dict[int, str] = {
    0: "#FF3333",
    1: "#FFD700",
    2: "#DDDDDD",
    3: "#33CC33",
    4: "#3399FF",
    5: "#888888",
}
ACTION_COLOR: dict[int, str] = {
    0: "#444444",
    1: "#FF3333",
    2: "#FFD700",
    3: "#DDDDDD",
}

# ---------------------------------------------------------------------------
# Model manifest
# ---------------------------------------------------------------------------


def _locate_manifest() -> Path:
    """Return the manifest path, trying multiple candidates in priority order.

    Priority:
    1. ``PITWALL_MANIFEST_PATH`` env var / Streamlit secret (explicit override)
    2. Repo-root-relative via ``__file__`` (works when CWD ≠ repo root)
    3. CWD-relative fallback (original behaviour, works when CWD = repo root)
    """
    env_override = os.getenv("PITWALL_MANIFEST_PATH")
    candidates: list[Path] = []
    if env_override:
        candidates.append(Path(env_override))
    candidates.append(Path(__file__).parent.parent.parent / "models_baked" / "manifest.json")
    candidates.append(Path("models_baked") / "manifest.json")
    for p in candidates:
        if p.exists():
            return p
    return candidates[1]  # return __file__-relative even if absent (error surfaces clearly)


def _load_manifest() -> dict[str, Any]:
    """Load the model catalog from models_baked/manifest.json.

    Falls back to a single hard-coded default entry when the file is absent,
    so the app still works on a fresh checkout before any manifest is written.
    """
    manifest_path = _locate_manifest()
    if manifest_path.exists():
        manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        return manifest
    return {
        "default": "default",
        "models": {
            "default": {
                "label": "PPO v1 (default)",
                "description": "Loaded from env vars / models_baked defaults.",
                "tire_uri": os.getenv("PITWALL_TIRE_MODEL_URI", "models_baked/tire"),
                "sc_uri": os.getenv("PITWALL_SC_MODEL_URI", "models_baked/sc"),
                "policy_path": os.getenv("PITWALL_POLICY_PATH", "models_baked/policy/model/policy"),
                "tags": {},
            }
        },
    }


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading models…")
def _load_models(
    model_key: str, tire_uri: str, sc_uri: str, policy_path: str, model_type: str
) -> tuple[Any, Any, Any]:
    """Load a model set by key.  Cached separately per unique (key, paths) triple.

    Args:
        model_key: Logical name from the manifest (used only as a cache key).
        tire_uri: Path or MLflow URI for the tire-degradation pyfunc model.
        sc_uri: Path or MLflow URI for the safety-car pyfunc model.
        policy_path: Path to the SB3 checkpoint (without .zip extension).
        model_type: ``"ppo"`` or ``"dqn"``.

    Returns:
        (tire_model, sc_model, policy) tuple.
    """
    tracking_uri = os.getenv("PITWALL_MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    tire = mlflow.pyfunc.load_model(tire_uri)
    sc = mlflow.pyfunc.load_model(sc_uri)
    loaded_policy: Any
    if model_type == "dqn":
        loaded_policy = DQNPER.load(
            policy_path,
            device="cpu",
            custom_objects={"replay_buffer_class": PrioritizedReplayBuffer},
        )
    else:
        loaded_policy = PPO.load(policy_path, device="cpu")
    return tire, sc, loaded_policy


def _recommend(
    obs: Any,
    policy: Any,
    model_type: str,
) -> tuple[int, str, dict[str, float], dict[str, float] | None]:
    """Dispatch recommend_action / probabilities to the right predict module.

    Returns:
        (action_int, action_label, probs_dict, q_values_dict_or_None)
    """
    if model_type == "dqn":
        action, label = _dqn_predict.recommend_action(obs, policy)
        probs = _dqn_predict.action_probabilities(obs, policy)
        qv: dict[str, float] | None = _dqn_predict.q_values(obs, policy)
    else:
        action, label = _ppo_predict.recommend_action(obs, policy)
        probs = _ppo_predict.action_probabilities(obs, policy)
        qv = None
    return action, label, probs, qv


@st.cache_data(show_spinner="Loading race data…")
def _load_gold() -> pd.DataFrame:
    return load_gold_seasons([2022, 2023, 2024, 2025])


@st.cache_resource(show_spinner=False)
def _get_seq_tire_model(model_type: str) -> tuple[Any, Any] | None:
    """Load and cache a PyTorch sequence tire model and its normalisation stats.

    Returns None when the artifact directory is absent (model not yet trained).

    Args:
        model_type: ``"tcn_gru"`` or ``"patch_tst"``.

    Returns:
        ``(nn_model, norm_stats)`` or ``None`` if artifacts are missing.
    """
    from typing import Literal

    if not _SEQ_MODELS_AVAILABLE or _load_seq_tire_model is None:
        return None
    mt: Literal["tcn_gru", "patch_tst"] = "patch_tst" if model_type == "patch_tst" else "tcn_gru"
    try:
        nn_model, norm_stats, _ = _load_seq_tire_model(mt)  # type: ignore[misc]
        return nn_model, norm_stats
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def _openf1_client() -> OpenF1Client:
    # 60s TTL halves the API request rate vs. the original 25s default, which
    # was triggering 429s during live sessions.  Laps are ~90s long so freshness
    # is fine for strategy decisions.
    #
    # PITWALL_OPENF1_API_KEY (or the bare OPENF1_API_KEY) unlocks OpenF1's
    # authenticated tier. During live sessions OpenF1 can restrict all API
    # access to authenticated users, so read both environment variables and
    # Streamlit secrets.
    api_key = os.getenv("PITWALL_OPENF1_API_KEY") or os.getenv("OPENF1_API_KEY")
    if not api_key:
        try:
            api_key = str(
                st.secrets.get("PITWALL_OPENF1_API_KEY")
                or st.secrets.get("OPENF1_API_KEY")
                or ""
            )
        except Exception:  # noqa: BLE001
            api_key = ""
    try:
        return OpenF1Client(ttl=60, api_key=api_key)
    except TypeError as exc:
        if "api_key" not in str(exc):
            raise
        client = OpenF1Client(ttl=60)
        if api_key and hasattr(client, "_session"):
            client._session.headers["Authorization"] = f"Bearer {api_key}"  # noqa: SLF001
        return client


def _client_is_rate_limited(client: OpenF1Client) -> bool:
    checker = getattr(client, "is_rate_limited", None)
    return bool(checker()) if callable(checker) else False


def _client_is_auth_required(client: OpenF1Client) -> bool:
    checker = getattr(client, "is_auth_required", None)
    return bool(checker()) if callable(checker) else False


@st.cache_resource(show_spinner="Connecting to S3…")
def _log_storage_status() -> tuple[bool, str]:
    """Probe S3/MinIO for log storage. Returns (ok, message).

    Cached so we don't re-probe on every Streamlit rerun.  Returns ``(True, bucket)``
    if the storage backend was reachable at startup; ``(False, error_message)``
    otherwise.  When unreachable, the Live tab disables the log toggle and the
    Prediction Log tab shows a setup hint.
    """
    try:
        storage = get_storage()
        return True, storage._bucket  # noqa: SLF001 — internal field used only for the status caption
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------


def _run_replay(
    race_df: pd.DataFrame,
    driver_num: int | str,
    tire_model: Any,
    sc_model: Any,
    policy: Any,
    model_type: str = "ppo",
) -> pd.DataFrame:
    """Replay the race and produce two position trajectories.

    Pass 1 follows what the team actually did so the AI can provide
    lap-by-lap commentary against a realistic race state.

    Pass 2 follows the AI's own recommendations from lap 1 to produce an
    honest counterfactual: "where would the driver have finished if Pitwall AI
    had full control of the strategy?"
    """

    def _apply_overrides(
        action: int,
        label: str,
        wet_frac: float,
        compound: int,
        tyre_life: int,
        cliff_lap: int,
        laps_left: int,
    ) -> tuple[int, str]:
        """Apply wet-weather and tyre-cliff overrides to a raw policy action."""
        on_wet = compound in (3, 4)
        if not on_wet:
            if wet_frac >= 0.8:
                return 5, "Pit — WET"
            if wet_frac >= 0.5:
                return 4, "Pit — INTER"
        elif on_wet and wet_frac < 0.2:
            return 2, "Pit — MEDIUM"
        if action == 0 and tyre_life >= cliff_lap + 3 and laps_left > 5:
            best = {0: 1, 1: 2, 2: 3}.get(compound, 2)
            return best, f"Pit — {['SOFT', 'MEDIUM', 'HARD'][best - 1]} ⚠"
        return action, label

    # ── Pass 1: team decisions → AI commentary ───────────────────────────────
    env = F1RaceEnv(race_df, driver_num, tire_model, sc_model)
    obs, _ = env.reset()
    agent_gold = race_df[race_df["driver_number"] == driver_num].sort_values("lap_number")
    rows: list[dict[str, Any]] = []
    terminated = False

    while not terminated:
        current_lap = env._lap
        raw_action, raw_label, probs, qv = _recommend(obs, policy, model_type)

        wet_frac = (
            float(env._wet_fraction_arr[current_lap]) if current_lap <= env._total_laps else 0.0
        )
        cliff = env._cliff_lap.get(env._compound, 35)
        laps_left = env._total_laps - current_lap
        action, label = _apply_overrides(
            raw_action, raw_label, wet_frac, env._compound, env._tyre_life, cliff, laps_left
        )

        gold_row = agent_gold[agent_gold["lap_number"] == current_lap]
        actual_pit = bool(gold_row["pit_in_this_lap"].iloc[0]) if not gold_row.empty else False
        actual_compound = (
            int(gold_row["compound_encoded"].iloc[0])
            if not gold_row.empty and pd.notna(gold_row["compound_encoded"].iloc[0])
            else env._compound
        )
        actual_position = (
            int(gold_row["position"].iloc[0])
            if not gold_row.empty and pd.notna(gold_row["position"].iloc[0])
            else None
        )
        team_action = {0: 1, 1: 2, 2: 3}.get(actual_compound, 1) if actual_pit else 0

        obs, reward, terminated, _, info = env.step(team_action)

        rows.append(
            {
                "lap": info["lap"],
                "actual_position": actual_position,
                "compound": info["compound"],
                "tyre_life": info["tyre_life"],
                "sim_lap_time_s": info["sim_lap_time_s"],
                "gap_ahead_s": info["gap_ahead_s"],
                "gap_behind_s": info["gap_behind_s"],
                "undercut_threat": info["undercut_threat"],
                "is_wet": info["is_wet"],
                "sc_probability": info["sc_probability"],
                "track_status": env._last_track_status,
                "recommended_action": action,
                "recommended_label": label,
                "pitted_this_lap": actual_pit,
                "prob_stay": probs["Stay out"],
                "prob_soft": probs["Pit — SOFT"],
                "prob_medium": probs["Pit — MEDIUM"],
                "prob_hard": probs["Pit — HARD"],
                "q_stay": qv["Stay out"] if qv else None,
                "q_soft": qv["Pit — SOFT"] if qv else None,
                "q_medium": qv["Pit — MEDIUM"] if qv else None,
                "q_hard": qv["Pit — HARD"] if qv else None,
                "actual_pit": actual_pit,
                "actual_compound": actual_compound,
                "reward": reward,
            }
        )

    df = pd.DataFrame(rows)

    # ── Pass 2: AI decisions → counterfactual positions ──────────────────────
    env2 = F1RaceEnv(race_df, driver_num, tire_model, sc_model)
    obs2, _ = env2.reset()
    terminated2 = False
    ai_pos_map: dict[int, int] = {}

    while not terminated2:
        current_lap2 = env2._lap
        raw_action2, raw_label2, _probs2, _qv2 = _recommend(obs2, policy, model_type)
        wet_frac2 = (
            float(env2._wet_fraction_arr[current_lap2]) if current_lap2 <= env2._total_laps else 0.0
        )
        cliff2 = env2._cliff_lap.get(env2._compound, 35)
        laps_left2 = env2._total_laps - current_lap2
        ai_action, _ = _apply_overrides(
            raw_action2, raw_label2, wet_frac2, env2._compound, env2._tyre_life, cliff2, laps_left2
        )
        # Actions 4/5 (INTER/WET override) are display-only; clamp to valid env range.
        ai_action = min(ai_action, 3)

        obs2, _, terminated2, _, info2 = env2.step(ai_action)
        ai_pos_map[info2["lap"]] = int(info2["position"])

    # Both the AI and team start from the same grid position.  Lap 1 position
    # from the sim can be off because the synthetic lap time doesn't exactly
    # match the real race lap 1 time.  Anchor to the gold-data starting position.
    lap1_rows = df[df["lap"] == 1]["actual_position"]
    if not lap1_rows.empty and pd.notna(lap1_rows.iloc[0]):
        ai_pos_map[1] = int(lap1_rows.iloc[0])

    df["ai_position"] = df["lap"].map(ai_pos_map)

    return df


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------


def _position_chart(replay: pd.DataFrame, total_laps: int) -> go.Figure:
    """Dual-line position chart: Pitwall AI counterfactual vs actual 2024 team result."""
    fig = go.Figure()

    # SC/VSC shading
    sc_laps = replay[replay["track_status"] >= 2]["lap"].tolist()
    for lap in sc_laps:
        fig.add_vrect(
            x0=lap - 0.5,
            x1=lap + 0.5,
            fillcolor="yellow",
            opacity=0.15,
            line_width=0,
        )

    # AI counterfactual: where the driver would be following Pitwall AI's calls
    ai = replay.dropna(subset=["ai_position"])
    fig.add_trace(
        go.Scatter(
            x=ai["lap"],
            y=ai["ai_position"],
            mode="lines+markers",
            name="Pitwall AI strategy",
            line={"color": "#E8002D", "width": 2},
            marker={"size": 4},
        )
    )

    # Actual position from gold data (what the team did)
    actual = replay.dropna(subset=["actual_position"])
    fig.add_trace(
        go.Scatter(
            x=actual["lap"],
            y=actual["actual_position"],
            mode="lines",
            name="Actual (2024)",
            line={"color": "#888888", "width": 1.5, "dash": "dot"},
        )
    )

    # Pit stop markers on the actual position line
    pits = replay[replay["pitted_this_lap"]].dropna(subset=["actual_position"])
    fig.add_trace(
        go.Scatter(
            x=pits["lap"],
            y=pits["actual_position"],
            mode="markers",
            name="Team pit stop",
            marker={"symbol": "triangle-down", "size": 12, "color": "#888888"},
            showlegend=True,
        )
    )

    # Laps where AI disagreed with team — mark on actual position line
    disagreed = replay[
        ((replay["recommended_action"] > 0) & ~replay["actual_pit"])
        | ((replay["recommended_action"] == 0) & replay["actual_pit"])
    ].dropna(subset=["actual_position"])
    fig.add_trace(
        go.Scatter(
            x=disagreed["lap"],
            y=disagreed["actual_position"],
            mode="markers",
            name="AI disagreed",
            marker={"symbol": "x", "size": 10, "color": "#FFD700"},
            showlegend=True,
        )
    )

    fig.update_layout(
        yaxis={"autorange": "reversed", "title": "Position", "dtick": 1},
        xaxis={"title": "Lap", "range": [1, total_laps]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 40, "r": 20, "t": 10, "b": 40},
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,1)",
        font={"color": "#FFFFFF"},
        yaxis_gridcolor="#333",
        xaxis_gridcolor="#333",
    )
    return fig


def _prob_bar(label: str, prob: float, color: str) -> None:
    cols = st.columns([3, 7])
    cols[0].caption(label)
    cols[1].progress(prob, text=f"{prob:.1%}")


# ---------------------------------------------------------------------------
# Live race tab
# ---------------------------------------------------------------------------

_PIT_LOSS_BY_CIRCUIT: dict[str, float] = {
    "Bahrain": 23.0,
    "Jeddah": 23.0,
    "Melbourne": 22.0,
    "Suzuka": 24.0,
    "Shanghai": 22.0,
    "Miami": 23.0,
    "Imola": 26.0,
    "Monaco": 25.0,
    "Montreal": 22.0,
    "Barcelona": 22.0,
    "Spielberg": 21.0,
    "Silverstone": 22.0,
    "Budapest": 22.0,
    "Spa-Francorchamps": 22.0,
    "Zandvoort": 19.0,
    "Monza": 19.0,
    "Baku": 24.0,
    "Singapore": 23.0,
    "Austin": 22.0,
    "Mexico City": 23.0,
    "Sao Paulo": 22.0,
    "Las Vegas": 23.0,
    "Losail": 23.0,
    "Abu Dhabi": 22.0,
}


def _is_race_type_session(s: dict[str, Any]) -> bool:
    """Return True for Race or Sprint sessions; False for qualifying/practice.

    OpenF1's ``session_type`` is "Race" for both feature races and sprints
    (the distinction is in ``session_name``).  Falls back to ``session_name``
    when ``session_type`` is missing.
    """
    if s.get("session_type") == "Race":
        return True
    return s.get("session_name") in ("Race", "Sprint")


def _pick_relevant_race_session(sessions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the race/sprint session closest in time to *now*.

    Naturally handles: currently-running session (smallest distance), just-finished
    session (small negative distance), about-to-start session (small positive
    distance).  Returns None when no race-type session exists in the input.
    """
    race_sessions = [s for s in sessions if _is_race_type_session(s)]
    if not race_sessions:
        return None
    now = datetime.now(timezone.utc)

    def _distance(s: dict[str, Any]) -> timedelta:
        ds = s.get("date_start")
        if not ds:
            return timedelta.max
        try:
            dt = datetime.fromisoformat(str(ds).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return timedelta.max
        return abs(dt - now)

    race_sessions.sort(key=_distance)
    return race_sessions[0]


def _format_gp_name(session_meta: dict[str, Any]) -> str:
    """Build a human-readable GP name with fallbacks.

    OpenF1 sometimes returns sessions without ``meeting_name`` populated.
    Fall back to ``country_name`` (e.g. "Canada GP") then ``circuit_short_name``.
    """
    name = session_meta.get("meeting_name")
    if name:
        return str(name)
    country = session_meta.get("country_name")
    if country:
        return f"{country} GP"
    circuit = session_meta.get("circuit_short_name")
    if circuit:
        return f"{circuit} GP"
    return "Unknown GP"


def _render_live_tab(policy: Any, model_type: str = "ppo", model_key: str = "default") -> None:
    """Render the Live Race tab — OpenF1 real-time strategy recommendations."""
    client = _openf1_client()

    # ── Session picker ───────────────────────────────────────────────────────
    col_sess, col_refresh, col_log = st.columns([3, 1, 1])
    with col_sess:
        use_latest = st.checkbox("Track latest session", value=True)
    with col_refresh:
        auto_refresh = st.toggle("Auto-refresh (5 s)", value=False)
    s3_ok, s3_msg = _log_storage_status()
    with col_log:
        log_enabled = st.toggle(
            "Log to S3",
            value=s3_ok,
            disabled=not s3_ok,
            help=(
                f"Persist one row per lap to s3://{s3_msg}/{LOG_PREFIX}/. "
                "Browse them in the Prediction Log tab."
                if s3_ok
                else f"S3 unreachable — logging disabled. {s3_msg}"
            ),
        )
    if not s3_ok:
        st.warning(
            f"⚠️ S3 storage unreachable, prediction logging is disabled.  "
            f"Set `PITWALL_STORAGE_BACKEND=s3`, `PITWALL_AWS_S3_BUCKET`, and AWS credentials "
            f"in your environment / Streamlit secrets to enable it.  Error: `{s3_msg}`"
        )

    if use_latest:
        # Filter to Race/Sprint sessions only — qualifying and practice produce
        # data shapes (no fixed lap count, no race-pace stints) that the PPO
        # policy isn't trained on, so the recommendation would be meaningless.
        year = datetime.now(timezone.utc).year
        try:
            sessions = client.get_sessions(year)
        except Exception as exc:
            st.error(f"OpenF1 API error: {exc}")
            return
        session_meta = _pick_relevant_race_session(sessions)

        # Fallback: OpenF1 moved /sessions?year=… behind authentication in late
        # 2025 (returns 401, treated as empty by _get).  /sessions?session_key=latest
        # is still public, so we can still show *something* — just without the
        # race/sprint type filter.
        if session_meta is None and not sessions:
            try:
                latest = client.get_latest_session()
            except Exception as exc:
                st.error(f"OpenF1 API error: {exc}")
                return
            if latest is not None:
                session_meta = latest
                st.caption(
                    "ℹ️ Race/sprint filter unavailable — OpenF1's `/sessions?year` endpoint "
                    "requires authentication.  Showing the latest session of any type. "
                    "Recommendations on qualifying/practice data are not meaningful."
                )

        if session_meta is None:
            if _client_is_auth_required(client):
                st.error(
                    "OpenF1 requires authentication while a live F1 session is in progress. "
                    "Add `PITWALL_OPENF1_API_KEY` or `OPENF1_API_KEY` to Streamlit secrets, "
                    "then reboot the app."
                )
            elif _client_is_rate_limited(client):
                st.info(
                    "⏳ OpenF1 API rate-limited — searching for the next race/sprint session…"
                )
            else:
                st.warning(
                    f"No race or sprint session found for {year} yet. "
                    "Toggle off 'Track latest session' to enter a specific session key."
                )
            if auto_refresh:
                time.sleep(5)
                st.rerun()
            elif st.button("🔄 Retry"):
                st.rerun()
            return
        session_key = int(session_meta["session_key"])
    else:
        session_key = st.number_input("Session key", value=9158, step=1)
        try:
            session_meta = client.get_session(int(session_key))
        except Exception as exc:
            st.error(f"OpenF1 API error: {exc}")
            return

    if session_meta is None:
        st.error(f"Session {session_key} not found.")
        return

    gp_name = _format_gp_name(session_meta)
    session_type = session_meta.get("session_name", "")
    total_laps = int(session_meta.get("total_laps") or 0)
    circuit_name = session_meta.get("circuit_short_name", "")
    pit_loss_s = _PIT_LOSS_BY_CIRCUIT.get(circuit_name, 22.0)

    st.markdown(f"### 🔴 {gp_name} — {session_type}")
    if total_laps == 0:
        st.warning("Session has no lap data yet (qualifying / practice / race not started).")

    # ── Driver picker ────────────────────────────────────────────────────────
    try:
        drivers = client.get_drivers(session_key)
    except Exception as exc:
        st.error(f"Failed to load drivers: {exc}")
        return

    if not drivers:
        if _client_is_auth_required(client):
            st.error(
                "OpenF1 requires authentication while this live session is in progress. "
                "Add `PITWALL_OPENF1_API_KEY` or `OPENF1_API_KEY` to Streamlit secrets, "
                "then reboot the app."
            )
        elif _client_is_rate_limited(client):
            st.info("⏳ OpenF1 API rate-limited — drivers will load shortly. Retrying automatically…")
        else:
            st.info("No drivers listed for this session yet.")
        # Keep the page alive: when auto-refresh is on, retry every 5s instead
        # of leaving the user stuck on an empty state with no rerun trigger.
        if auto_refresh:
            time.sleep(5)
            st.rerun()
        elif st.button("🔄 Retry"):
            st.rerun()
        return

    driver_map = {
        f"#{d['driver_number']}  {d.get('name_acronym', '')}  · {d.get('team_name', '')}": d[
            "driver_number"
        ]
        for d in sorted(drivers, key=lambda x: x.get("driver_number", 99))
    }
    chosen_label = st.selectbox("Driver", list(driver_map.keys()))
    driver_number = int(driver_map[chosen_label])

    # State key per (session, driver)
    state_key = f"live_state_{session_key}_{driver_number}"
    session_matches = st.session_state.get(f"{state_key}_session") == session_key
    if state_key not in st.session_state or not session_matches:
        st.session_state[state_key] = DriverLiveState(
            driver_number=driver_number,
            total_laps=max(total_laps, 1),
            pit_loss_s=pit_loss_s,
            n_drivers=len(drivers),
        )
        st.session_state[f"{state_key}_session"] = session_key

    live_state: DriverLiveState = st.session_state[state_key]

    # ── Fetch & update state ─────────────────────────────────────────────────
    try:
        lap_record = client.get_latest_lap(session_key, driver_number)
        stint_record = client.get_current_stint(session_key, driver_number, live_state.current_lap)
        position = client.get_latest_position(session_key, driver_number)
        pit_stops = len(client.get_pit_stops(session_key, driver_number))
        sc_active = client.is_safety_car_active(session_key, live_state.current_lap)
        field_stints = client.get_stints(session_key)
        weather = client.get_latest_weather(session_key)
    except Exception as exc:
        st.warning(f"Live data fetch failed: {exc}")
        lap_record = None
        stint_record = None
        position = None
        pit_stops = 0
        sc_active = False
        field_stints = []
        weather = None

    update_from_openf1(
        live_state,
        lap_record,
        stint_record,
        position,
        pit_stops,
        sc_active,
        field_stints,
        weather,
    )
    st.session_state[state_key] = live_state

    obs = live_state.build_obs()

    # ── Policy recommendation ────────────────────────────────────────────────
    action, label, probs, qv = _recommend(obs, policy, model_type)

    # Wet-weather override (mirrors replay engine rule)
    on_wet = live_state.compound_encoded in (3, 4)
    if not on_wet:
        if live_state.wet_fraction >= 0.8:
            action, label = 5, "Pit — WET"
        elif live_state.wet_fraction >= 0.5:
            action, label = 4, "Pit — INTER"
    elif on_wet and live_state.wet_fraction < 0.2:
        action, label = 2, "Pit — MEDIUM"

    # Cliff override
    cliff = live_state.cliff_lap()
    laps_left = max(live_state.total_laps - live_state.current_lap, 0)
    if action == 0 and live_state.tyre_life >= cliff + 3 and laps_left > 5:
        best = {0: 1, 1: 2, 2: 3}.get(live_state.compound_encoded, 2)
        action, label = best, f"Pit — {['SOFT', 'MEDIUM', 'HARD'][best - 1]} ⚠"

    # ── Persist prediction (append-on-lap-change) ────────────────────────────
    log_status_msg: str | None = None
    if log_enabled and live_state.current_lap > 0:
        last_logged_key = f"last_logged_lap_{session_key}_{driver_number}"
        last_logged = st.session_state.get(last_logged_key, 0)
        if live_state.current_lap > last_logged:
            log_year = race_year(session_meta)
            log_race = race_folder_name(session_meta)
            try:
                append_prediction(
                    PredictionRow(
                        session_key=int(session_key),
                        session_name=str(session_type),
                        gp_name=str(gp_name),
                        circuit=str(circuit_name),
                        driver_number=int(driver_number),
                        driver_label=str(chosen_label),
                        model_key=str(model_key),
                        model_type=str(model_type),
                        state=live_state,
                        obs=obs,
                        recommended_action=int(action),
                        recommended_label=str(label),
                        probs=probs,
                        q_values=qv,
                    ),
                    year=log_year,
                    race_folder=log_race,
                )
                st.session_state[last_logged_key] = live_state.current_lap
                log_status_msg = f"📝 Logged lap {live_state.current_lap} → {log_year}/{log_race}"
            except Exception as exc:  # noqa: BLE001
                log_status_msg = f"⚠️ Log write failed: {exc}"

    # ── Dashboard ────────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Lap", f"{live_state.current_lap} / {live_state.total_laps}")
    m2.metric("Position", f"P{live_state.position}")
    m3.metric("Tyre age", f"{live_state.tyre_life} laps")
    compound_name = {0: "SOFT", 1: "MEDIUM", 2: "HARD", 3: "INTER", 4: "WET", 5: "?"}[
        live_state.compound_encoded
    ]
    m4.metric("Compound", compound_name)
    m5.metric("Pit stops", live_state.pit_stops)

    st.divider()

    rec_col, flags_col = st.columns([1, 1])

    with rec_col:
        st.markdown("#### Pitwall AI recommendation")
        rec_color = ACTION_COLOR.get(min(action, 3), "#888")
        st.markdown(
            f'<h3 style="color:{rec_color}">{label}</h3>',
            unsafe_allow_html=True,
        )
        _prob_bar("Stay out", probs["Stay out"], "#444")
        _prob_bar("Pit — SOFT", probs["Pit — SOFT"], "#FF3333")
        _prob_bar("Pit — MEDIUM", probs["Pit — MEDIUM"], "#FFD700")
        _prob_bar("Pit — HARD", probs["Pit — HARD"], "#DDDDDD")
        if qv is not None:
            st.caption("Q-values (expected cumulative reward per action)")
            for act_label, q in qv.items():
                marker = " ◀" if act_label == label else ""
                st.caption(f"`{act_label}`: **{q:+.3f}**{marker}")

    with flags_col:
        st.markdown("#### Race flags")
        if sc_active:
            st.error("🟡 Safety Car / VSC active — free stop opportunity!")
        if live_state.wet_fraction >= 0.5:
            st.warning(f"🌧️ Wet conditions — {live_state.wet_fraction:.0%} of field on wet tyres")
        if live_state.tyre_life >= cliff:
            st.warning(f"⚠️ Tyre cliff approaching (cliff ≈ lap {cliff})")
        if live_state.gap_behind_s < 3.0 and live_state.tyre_age_behind < live_state.tyre_life:
            st.warning("🔥 Undercut threat from behind")
        if weather is not None:
            track_temp = weather.get("track_temperature")
            if track_temp is not None:
                st.caption(f"Track temp: {track_temp} °C")

    # ── Auto-refresh ─────────────────────────────────────────────────────────
    ts = time.strftime("%H:%M:%S")
    footer = f"Last updated: {ts}  ·  Session key: {session_key}"
    if log_status_msg:
        footer = f"{footer}  ·  {log_status_msg}"
    if _client_is_rate_limited(client):
        footer = f"{footer}  ·  ⏳ OpenF1 rate-limited — serving cached data"
    if _client_is_auth_required(client):
        footer = f"{footer}  ·  OpenF1 auth required for live data"
    st.caption(footer)

    if auto_refresh:
        # 5s rerun cadence; OpenF1Client TTL is 60s so most calls hit cache
        # and the effective API rate stays at ~1 call per minute per endpoint.
        time.sleep(5)
        st.rerun()
    else:
        if st.button("🔄 Refresh now"):
            st.rerun()


# ---------------------------------------------------------------------------
# Tire degradation forecast
# ---------------------------------------------------------------------------

_OBSERVE_LAPS: int = 5  # laps shown as observed history before the forecast horizon


def _tire_forecast_chart(
    driver_df: pd.DataFrame,
    tire_type: str,
    lgbm_tire_model: Any,
    seq_assets: tuple[Any, Any] | None,
) -> go.Figure:
    """Plot actual vs model-predicted tire degradation for the longest stint.

    Uses the first ``_OBSERVE_LAPS`` laps as observed history; the model
    forecasts all remaining laps autoregressively.  For LightGBM the scalar
    :func:`predict_stint_degradation` interface is used; for sequence models
    :func:`predict_stint_from_history` is used.

    Args:
        driver_df: Gold DataFrame filtered to one driver and one race.
        tire_type: ``"lgbm"``, ``"tcn_gru"``, or ``"patch_tst"``.
        lgbm_tire_model: Loaded MLflow pyfunc LightGBM model (always available).
        seq_assets: ``(nn_model, norm_stats)`` for sequence models, or None.

    Returns:
        Plotly Figure (empty data list if not enough laps to plot).
    """
    fig = go.Figure()

    if "stint_number" not in driver_df.columns or driver_df["stint_number"].isna().all():
        return fig

    stint_counts = (
        driver_df.dropna(subset=["stint_number"]).groupby("stint_number")["lap_number"].count()
    )
    if stint_counts.empty:
        return fig

    longest_stint = int(stint_counts.idxmax())
    stint_df = (
        driver_df[driver_df["stint_number"] == longest_stint]
        .sort_values("lap_number")
        .dropna(subset=["lap_time_delta_s", "tyre_life_laps"])
        .copy()
    )
    # Fill columns required by sequence models.
    stint_df["is_fresh_tyre"] = stint_df["is_fresh_tyre"].fillna(False).astype(float)
    stint_df["tyre_deg_rate_s_per_lap"] = stint_df["tyre_deg_rate_s_per_lap"].fillna(0.0)
    stint_df["sc_laps_since_last"] = stint_df["sc_laps_since_last"].fillna(50.0)
    stint_df["position"] = stint_df["position"].fillna(10.0)

    if len(stint_df) < _OBSERVE_LAPS + 1:
        return fig

    tyre_life = stint_df["tyre_life_laps"].tolist()
    actual_delta = stint_df["lap_time_delta_s"].tolist()
    n_forecast = len(stint_df) - _OBSERVE_LAPS
    forecast_tyre_life = tyre_life[_OBSERVE_LAPS:]

    # Actual scatter
    fig.add_trace(
        go.Scatter(
            x=tyre_life,
            y=actual_delta,
            mode="markers",
            name="Actual",
            marker={"color": "#aaaaaa", "size": 6, "symbol": "circle"},
        )
    )

    # Vertical split between observed and forecast
    split_x = tyre_life[_OBSERVE_LAPS - 1]
    fig.add_vline(
        x=split_x,
        line_dash="dot",
        line_color="#555555",
        annotation_text=f"← {_OBSERVE_LAPS} obs | forecast →",
        annotation_position="top",
    )

    # Model forecast
    observed_df = stint_df.head(_OBSERVE_LAPS)
    compound_enc = int(stint_df["compound_encoded"].iloc[0])
    color = _TIRE_FORECAST_COLORS.get(tire_type, "#ffffff")
    model_label = next((k for k, v in _TIRE_FORECAST_OPTIONS.items() if v == tire_type), tire_type)

    preds: list[float] = []
    if tire_type == "lgbm":
        rp_start = float(observed_df["race_progress"].iloc[-1])
        baseline_delta = float(observed_df["lap_delta_to_field_median_s"].mean())
        raw = _lgbm_stint_predict(
            n_forecast, compound_enc, rp_start, baseline_delta, lgbm_tire_model
        )
        preds = list(raw)
    elif seq_assets is not None:
        nn_model, norm_stats = seq_assets
        history = observed_df[_SEQ_FEATURE_COLS].copy()
        raw_seq = _seq_stint_forecast(history, n_forecast, nn_model, norm_stats)
        preds = list(raw_seq)

    if preds:
        fig.add_trace(
            go.Scatter(
                x=forecast_tyre_life,
                y=preds,
                mode="lines+markers",
                name=f"{model_label}",
                line={"color": color, "width": 2},
                marker={"symbol": "diamond", "size": 5},
            )
        )

    compound_name = COMPOUND_LABEL.get(compound_enc, "?")
    fig.update_layout(
        title=f"Tire Degradation — Stint {longest_stint} ({compound_name})",
        xaxis_title="Tyre Age (laps)",
        yaxis_title="Lap Time Delta (s vs driver stint median)",
        template="plotly_dark",
        height=350,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


# ---------------------------------------------------------------------------
# Prediction Log tab
# ---------------------------------------------------------------------------


def _render_log_tab() -> None:
    """Render the Prediction Log tab — browse persisted race logs by year/race."""
    st.markdown("### 📜 Online Prediction Log")

    s3_ok, s3_msg = _log_storage_status()
    if not s3_ok:
        st.error(
            f"⚠️ S3 storage unreachable, can't list logs.  "
            f"Set `PITWALL_STORAGE_BACKEND=s3`, `PITWALL_AWS_S3_BUCKET`, and AWS credentials "
            f"in your environment / Streamlit secrets.  Error: `{s3_msg}`"
        )
        return

    st.caption(
        f"Persisted lap-by-lap predictions live under "
        f"`s3://{s3_msg}/{LOG_PREFIX}/<year>/<race>/<driver>.parquet`.  "
        "Pick a race below to inspect every driver Pitwall AI watched during it."
    )

    races = list_races()
    if not races:
        st.info(
            "No logs yet.  Open the **🔴 Live Race** tab during a session (with **Log to S3** "
            "enabled) and the lap-by-lap recommendations will be captured here."
        )
        return

    # ── Year / race pickers ──────────────────────────────────────────────────
    years = sorted({r["year"] for r in races}, reverse=True)
    col_year, col_race = st.columns([1, 3])
    with col_year:
        selected_year = st.selectbox("Year", years, key="log_year")
    year_races = [r for r in races if r["year"] == selected_year]
    race_options = {
        f"{r['race_folder']}  ·  {r['n_drivers']} driver(s)": r["race_folder"]
        for r in year_races
    }
    with col_race:
        selected_race_label = st.selectbox("Race", list(race_options.keys()), key="log_race")
    selected_race = race_options[selected_race_label]

    df = load_race(selected_year, selected_race)
    if df.empty:
        st.warning("Race folder exists in S3 but no rows could be loaded.")
        return

    # ── Summary metrics across the whole race ────────────────────────────────
    n_drivers = int(df["driver_number"].nunique())
    n_rows = int(len(df))
    n_laps_max = int(df["lap"].max()) if not df.empty else 0
    n_pits = int((df["recommended_action"] > 0).sum())
    last_row = df.sort_values("timestamp_utc").iloc[-1]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Drivers", n_drivers)
    m2.metric("Total laps logged", n_rows)
    m3.metric("Max lap", n_laps_max)
    m4.metric("Pit recs", n_pits)
    m5.metric("Model", f"{last_row.get('model_key', '?')} ({last_row.get('model_type', '?')})")

    # ── Driver filter ────────────────────────────────────────────────────────
    driver_labels = (
        df.sort_values("driver_number")
        .groupby("driver_number")["driver_label"]
        .last()
        .to_dict()
    )
    driver_choices = ["All drivers"] + [
        f"#{num}  {label}" for num, label in driver_labels.items()
    ]
    selected_driver_label = st.selectbox("Driver filter", driver_choices, key="log_driver")
    if selected_driver_label == "All drivers":
        view_df = df
    else:
        picked_num = int(selected_driver_label.split()[0].lstrip("#"))
        view_df = df[df["driver_number"] == picked_num]

    # ── Position chart: one line per driver in the view ──────────────────────
    fig = go.Figure()
    palette = [
        "#E8002D", "#1E5BC6", "#27F4D2", "#FF8000", "#52E252", "#FFD700",
        "#3671C6", "#B6BABD", "#6692FF", "#229971", "#37BEDD", "#F58020",
        "#358C75", "#A6051A", "#B6BABD", "#27F4D2", "#5E8FAA", "#52E252",
        "#FF8000", "#229971",
    ]
    for idx, (num, group) in enumerate(view_df.sort_values("lap").groupby("driver_number")):
        label = driver_labels.get(num, f"#{num}")
        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=group["lap"],
                y=group["position"],
                mode="lines+markers",
                name=str(label),
                line={"color": color, "width": 2},
                marker={"size": 4},
            )
        )
        pit_recs = group[group["recommended_action"] > 0]
        if not pit_recs.empty:
            fig.add_trace(
                go.Scatter(
                    x=pit_recs["lap"],
                    y=pit_recs["position"],
                    mode="markers",
                    name=f"{label} · pit rec",
                    marker={"symbol": "triangle-down", "size": 10, "color": color},
                    showlegend=False,
                )
            )
    sc_laps = view_df[view_df["sc_active"]]["lap"].unique().tolist()
    for lap in sc_laps:
        fig.add_vrect(
            x0=lap - 0.5, x1=lap + 0.5, fillcolor="yellow", opacity=0.10, line_width=0
        )
    fig.update_layout(
        yaxis={"autorange": "reversed", "title": "Position", "dtick": 1},
        xaxis={"title": "Lap"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 40, "r": 20, "t": 10, "b": 40},
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,1)",
        font={"color": "#FFFFFF"},
        yaxis_gridcolor="#333",
        xaxis_gridcolor="#333",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "🟡 Yellow bands = SC/VSC active when prediction was made  ·  ▼ = pit recommendation per driver"
    )

    # ── Full log table + download ────────────────────────────────────────────
    view_cols = [
        "timestamp_utc",
        "driver_label",
        "lap",
        "position",
        "compound_name",
        "tyre_life",
        "pit_stops",
        "sc_active",
        "wet_fraction",
        "recommended_label",
        "prob_stay",
        "prob_soft",
        "prob_medium",
        "prob_hard",
    ]
    table = view_df[view_cols].sort_values(["driver_label", "lap"]).copy()
    for col in ("prob_stay", "prob_soft", "prob_medium", "prob_hard"):
        table[col] = table[col].apply(lambda x: f"{x:.1%}")
    table["wet_fraction"] = table["wet_fraction"].apply(lambda x: f"{x:.0%}")
    st.dataframe(table, use_container_width=True, hide_index=True)

    csv_bytes = view_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇ Download {selected_year}/{selected_race} (CSV)",
        data=csv_bytes,
        file_name=f"pitwall_log_{selected_year}_{selected_race}.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Pitwall AI — F1 Strategy Engine",
        page_icon="🏎️",
        layout="wide",
    )

    # Custom CSS — dark F1 feel
    st.markdown(
        """
    <style>
        .block-container { padding-top: 1rem; }
        .metric-label { font-size: 0.75rem; color: #999; }
        .compound-badge {
            display: inline-block; padding: 2px 10px;
            border-radius: 12px; font-weight: 700; font-size: 0.85rem;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ── Sidebar — model selector (applies to both tabs) ──────────────────────
    manifest = _load_manifest()
    model_catalog: dict[str, Any] = manifest["models"]
    default_key: str = manifest.get("default", next(iter(model_catalog)))

    with st.sidebar:
        st.title("🏎️ Pitwall AI")
        st.caption("F1 Race Strategy Engine")
        st.divider()

        # Model selector
        model_labels = {v["label"]: k for k, v in model_catalog.items()}
        default_label = model_catalog[default_key]["label"]
        selected_model_label = st.selectbox(
            "Strategy model",
            list(model_labels.keys()),
            index=list(model_labels.keys()).index(default_label),
        )
        selected_model_key = model_labels[selected_model_label]
        selected_model_meta = model_catalog[selected_model_key]

        algo_badge = "🧠 DQN+PER" if selected_model_meta.get("model_type") == "dqn" else "📈 PPO"
        st.caption(f"Active: **{algo_badge}** — {selected_model_label}")

        with st.expander("Model details", expanded=False):
            st.caption(selected_model_meta.get("description", ""))
            tags = selected_model_meta.get("tags", {})
            for tag_k, tag_v in tags.items():
                st.caption(f"**{tag_k}:** {tag_v}")

        st.divider()
        selected_tire_label = st.selectbox(
            "Tire model",
            list(_TIRE_FORECAST_OPTIONS.keys()),
            key="tire_forecast_model",
            help=(
                "Tire degradation model used in the race simulator and the "
                "forecast panel. LightGBM uses point-in-time features; "
                "TCN+GRU / PatchTST use a 15-lap history window."
            ),
        )
        selected_tire_type = _TIRE_FORECAST_OPTIONS[selected_tire_label]

    # ── Load the selected model set (cached per unique paths) ────────────────
    model_type: str = selected_model_meta.get("model_type", "ppo")
    tire_model, sc_model, policy = _load_models(
        model_key=selected_model_key,
        tire_uri=selected_model_meta["tire_uri"],
        sc_uri=selected_model_meta["sc_uri"],
        policy_path=selected_model_meta["policy_path"],
        model_type=model_type,
    )

    # Load sequence tire model when needed (cached per model type).
    if not _SEQ_MODELS_AVAILABLE and _seq_import_error:
        st.sidebar.error(f"Sequence models unavailable — import failed:\n```\n{_seq_import_error}\n```")
    seq_tire_assets: tuple[Any, Any] | None = None
    if selected_tire_type != "lgbm":
        seq_tire_assets = _get_seq_tire_model(selected_tire_type)
        if seq_tire_assets is None:
            st.sidebar.warning(
                f"Sequence tire model `{selected_tire_type}` not found. "
                "Run `make train-tire-tcn` or `make train-tire-pst` first."
            )

    # Use the sequence model in the simulator when one is selected and available.
    # SequenceTireWrapper exposes the same .predict(df) interface as the LightGBM pyfunc.
    effective_tire_model = (
        SequenceTireWrapper(seq_tire_assets[0], seq_tire_assets[1])  # type: ignore[misc]
        if (seq_tire_assets is not None and SequenceTireWrapper is not None)
        else tire_model
    )

    st.info(
        "**📊 Race Replay** — replay any race from 2022–2025 and see what Pitwall AI would have "
        "recommended lap-by-lap.  |  **🔴 Live Race** — real-time recommendations via the OpenF1 "
        "API during a race weekend.  |  **📜 Prediction Log** — browse persisted lap-by-lap "
        "recommendations captured during live sessions.",
        icon="ℹ️",
    )
    tab_replay, tab_live, tab_log = st.tabs(
        ["📊 Race Replay", "🔴 Live Race", "📜 Prediction Log"]
    )

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1 — Race Replay
    # ═════════════════════════════════════════════════════════════════════════
    with tab_replay:
        gold = _load_gold()

        season_col, rnd_col = st.columns([1, 3])
        with season_col:
            available_seasons = sorted(gold["year"].unique().tolist(), reverse=True)
            selected_season = st.selectbox(
                "Season",
                available_seasons,
                index=available_seasons.index(2024) if 2024 in available_seasons else 0,
                key="replay_season",
            )

        season_gold = gold[(gold["year"] == selected_season) & (gold["session"] == "R")]
        year_names = ROUND_NAMES_BY_YEAR.get(selected_season, {})
        available_rounds = sorted(int(r) for r in season_gold["round_number"].unique())
        round_options = {
            f"Rnd {r:02d} — {year_names.get(r, f'Round {r}')}": r for r in available_rounds
        }

        default_rnd_idx = min(9, len(round_options) - 1)
        with rnd_col:
            selected_label = st.selectbox(
                "Race", list(round_options.keys()), index=default_rnd_idx, key="replay_round"
            )
        selected_round = round_options[selected_label]

        race_df = season_gold[season_gold["round_number"] == selected_round].copy()

        if race_df.empty:
            st.error(
                f"No race data found for {selected_season} Round {selected_round}. "
                "Run `make ingest-season` first."
            )
        else:
            # Driver selector
            driver_options: dict[str, int | str] = {}
            for drv, grp in race_df.groupby("driver_number"):
                abbr = (
                    grp["driver_code"].iloc[0]
                    if "driver_code" in grp.columns and pd.notna(grp["driver_code"].iloc[0])
                    else str(drv)
                )
                team = (
                    grp["team"].iloc[0]
                    if "team" in grp.columns and pd.notna(grp["team"].iloc[0])
                    else ""
                )
                lbl = f"#{drv}  {abbr}" + (f"  · {team}" if team else "")
                driver_options[lbl] = cast(int, drv)

            selected_driver_label = st.selectbox("Driver", list(driver_options.keys()))
            selected_driver = driver_options[selected_driver_label]

            # Auto-run whenever season, race, driver, or tire model changes.
            cache_key = (
                f"{selected_model_key}_{selected_tire_type}_"
                f"{selected_season}_{selected_round}_{selected_driver}"
            )
            if st.session_state.get("replay_key") != cache_key:
                with st.spinner("Simulating race…"):
                    st.session_state.replay = _run_replay(
                        race_df,
                        selected_driver,
                        effective_tire_model,
                        sc_model,
                        policy,
                        model_type,
                    )
                    st.session_state.replay_key = cache_key

            replay: pd.DataFrame = st.session_state.get("replay", pd.DataFrame())
            if replay.empty:
                st.info("Simulation loading…")
            else:
                total_laps = int(race_df["lap_number"].max())
                event_name = year_names.get(selected_round, f"Round {selected_round}")

                # ── Header ────────────────────────────────────────────────────
                parts = selected_driver_label.split()
                driver_abbr = parts[1] if len(parts) > 1 else str(selected_driver)
                st.markdown(
                    f"## {event_name}  ·  {selected_season}  ·  #{selected_driver} {driver_abbr}"
                )

                # Summary metrics
                final = replay.iloc[-1]
                ai_pos = int(final["ai_position"]) if pd.notna(final["ai_position"]) else "—"
                actual_pos = (
                    int(final["actual_position"]) if pd.notna(final["actual_position"]) else "—"
                )
                n_pits = int(replay["pitted_this_lap"].sum())

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Pitwall AI finish", f"P{ai_pos}")
                m2.metric("Actual finish", f"P{actual_pos}")
                m3.metric("AI pit stops", n_pits)
                m4.metric("Total laps", total_laps)

                # ── Position chart ────────────────────────────────────────────
                st.plotly_chart(_position_chart(replay, total_laps), use_container_width=True)
                st.caption("🟡 Yellow bands = Safety Car / VSC period  ·  ▼ = Pit stop")

                st.divider()

                # ── Lap explorer ──────────────────────────────────────────────
                st.subheader("Lap-by-lap explorer")
                selected_lap = st.slider("Lap", 1, len(replay), 1)
                row_df = replay[replay["lap"] == selected_lap]
                if row_df.empty:
                    row_df = replay.iloc[selected_lap - 1 : selected_lap]
                row = row_df.iloc[0]

                left, right = st.columns(2)

                with left:
                    st.markdown("#### Race state")
                    compound = int(row["compound"])
                    comp_color = COMPOUND_COLOR.get(compound, "#888")
                    comp_name = COMPOUND_LABEL.get(compound, "?")
                    st.markdown(
                        f'<span class="compound-badge" style="background:{comp_color};'
                        f'color:{"#111" if compound in (1, 2, 3) else "#fff"}">'
                        f"{comp_name}</span>",
                        unsafe_allow_html=True,
                    )
                    c1, c2 = st.columns(2)
                    c1.metric("Tyre age", f"{int(row['tyre_life'])} laps")
                    actual_pos_lap = (
                        int(row["actual_position"]) if pd.notna(row["actual_position"]) else "?"
                    )
                    c2.metric("Position (actual)", f"P{actual_pos_lap}")

                    gap_ahead = row["gap_ahead_s"]
                    gap_behind = row["gap_behind_s"]
                    c3, c4 = st.columns(2)
                    c3.metric("Gap ahead", f"{gap_ahead:.1f}s" if gap_ahead < 59 else "—")
                    c4.metric("Gap behind", f"{gap_behind:.1f}s" if gap_behind < 59 else "—")

                    flags: list[str] = []
                    if row["undercut_threat"]:
                        flags.append("⚠️ **Undercut threat**")
                    if row["track_status"] >= 2:
                        flags.append("🟡 **Safety Car / VSC**")
                    if row["is_wet"]:
                        flags.append("🌧️ **Wet conditions**")
                    if row["sc_probability"] > 0.3:
                        flags.append(f"⚡ SC risk {row['sc_probability']:.0%}")
                    if flags:
                        st.markdown("  ".join(flags))

                with right:
                    st.markdown("#### Pitwall AI recommendation")
                    action = int(row["recommended_action"])
                    rec_color = ACTION_COLOR.get(action, "#888")
                    st.markdown(
                        f'<h3 style="color:{rec_color}">{row["recommended_label"]}</h3>',
                        unsafe_allow_html=True,
                    )

                    _prob_bar("Stay out", row["prob_stay"], "#444")
                    _prob_bar("Pit — SOFT", row["prob_soft"], "#FF3333")
                    _prob_bar("Pit — MEDIUM", row["prob_medium"], "#FFD700")
                    _prob_bar("Pit — HARD", row["prob_hard"], "#DDDDDD")

                    if model_type == "dqn" and pd.notna(row.get("q_stay")):
                        st.caption("Q-values (expected cumulative reward per action)")
                        q_map = {
                            "Stay out": row["q_stay"],
                            "Pit — SOFT": row["q_soft"],
                            "Pit — MEDIUM": row["q_medium"],
                            "Pit — HARD": row["q_hard"],
                        }
                        best_q = max(q_map.values())
                        for act_label, q in q_map.items():
                            marker = " ◀" if q == best_q else ""
                            st.caption(f"`{act_label}`: **{q:+.3f}**{marker}")

                    total_pit_prob = row["prob_soft"] + row["prob_medium"] + row["prob_hard"]
                    if action == 0 and total_pit_prob > 0.25:
                        st.warning(
                            f"Pit window opening — {total_pit_prob:.0%} chance pit is optimal"
                        )

                    st.divider()
                    actual_decision = "Pit" if row["actual_pit"] else "Stay out"
                    agreed = (action == 0 and not row["actual_pit"]) or (
                        action > 0 and row["actual_pit"]
                    )
                    icon = "✅" if agreed else "⚠️"
                    st.markdown(f"**Actual team decision:** {actual_decision}  {icon}")
                    if not agreed:
                        if action > 0 and not row["actual_pit"]:
                            st.caption("Pitwall AI would pit here — team stayed out.")
                        else:
                            st.caption("Team pitted — Pitwall AI would stay out.")

                with st.expander("Full race log"):
                    display = replay[
                        [
                            "lap",
                            "ai_position",
                            "actual_position",
                            "compound",
                            "tyre_life",
                            "sim_lap_time_s",
                            "gap_ahead_s",
                            "gap_behind_s",
                            "recommended_label",
                            "recommended_action",
                            "actual_pit",
                        ]
                    ].copy()
                    display["compound"] = display["compound"].map(COMPOUND_LABEL)
                    display["gap_ahead_s"] = display["gap_ahead_s"].apply(
                        lambda x: f"{x:.1f}s" if x < 59 else "—"
                    )
                    display["gap_behind_s"] = display["gap_behind_s"].apply(
                        lambda x: f"{x:.1f}s" if x < 59 else "—"
                    )
                    display["ai_would_pit"] = display["recommended_action"] > 0
                    display.drop(columns=["recommended_action"], inplace=True)
                    display.columns = [
                        "Lap",
                        "AI Pos",
                        "Actual Pos",
                        "Compound",
                        "Tyre Age",
                        "Sim Lap (s)",
                        "Gap Ahead",
                        "Gap Behind",
                        "AI Recommendation",
                        "Team Pitted",
                        "AI Would Pit",
                    ]
                    st.dataframe(display, use_container_width=True, hide_index=True)

                st.divider()
                with st.expander("🔬 Tire Degradation Forecast", expanded=True):
                    driver_mask = race_df["driver_number"] == str(selected_driver)
                    if not driver_mask.any():
                        driver_mask = race_df["driver_number"] == selected_driver
                    driver_race_df = race_df[driver_mask].copy()

                    if driver_race_df.empty:
                        st.info("No gold data for this driver.")
                    else:
                        fig_tire = _tire_forecast_chart(
                            driver_race_df,
                            selected_tire_type,
                            tire_model,
                            seq_tire_assets,
                        )
                        if fig_tire.data:
                            st.plotly_chart(fig_tire, use_container_width=True)
                            st.caption(
                                f"Grey dots = actual recorded deltas from gold data.  "
                                f"First **{_OBSERVE_LAPS} laps** used as observed history; "
                                f"**{selected_tire_label}** forecasts the remainder "
                                "autoregressively.  "
                                "Switch the tire model in the sidebar to compare architectures."
                            )
                        else:
                            st.info(
                                "Not enough stint data to plot (need at least "
                                f"{_OBSERVE_LAPS + 1} clean laps in one stint)."
                            )

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2 — Live Race
    # ═════════════════════════════════════════════════════════════════════════
    with tab_live:
        _render_live_tab(policy, model_type, selected_model_key)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3 — Prediction Log
    # ═════════════════════════════════════════════════════════════════════════
    with tab_log:
        _render_log_tab()


if __name__ == "__main__":
    main()
