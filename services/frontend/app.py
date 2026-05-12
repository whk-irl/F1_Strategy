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
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ml.models._loader import load_gold_seasons
from ml.models.strategy_policy.predict import (
    action_probabilities,
    recommend_action,
)
from stable_baselines3 import PPO

from services.live.obs_builder import DriverLiveState, update_from_openf1
from services.live.openf1_client import OpenF1Client
from services.simulator.env import F1RaceEnv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUND_NAMES: dict[int, str] = {
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

_MANIFEST_PATH = Path("models_baked/manifest.json")


@st.cache_data(show_spinner=False)
def _load_manifest() -> dict[str, Any]:
    """Load the model catalog from models_baked/manifest.json.

    Falls back to a single hard-coded default entry when the file is absent,
    so the app still works on a fresh checkout before any manifest is written.
    """
    if _MANIFEST_PATH.exists():
        manifest: dict[str, Any] = json.loads(_MANIFEST_PATH.read_text())
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
    model_key: str, tire_uri: str, sc_uri: str, policy_path: str
) -> tuple[Any, Any, Any]:
    """Load a model set by key.  Cached separately per unique (key, paths) triple.

    Args:
        model_key: Logical name from the manifest (used only as a cache key).
        tire_uri: Path or MLflow URI for the tire-degradation pyfunc model.
        sc_uri: Path or MLflow URI for the safety-car pyfunc model.
        policy_path: Path to the SB3 PPO checkpoint (without .zip extension).

    Returns:
        (tire_model, sc_model, ppo_policy) tuple.
    """
    tracking_uri = os.getenv("PITWALL_MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    tire = mlflow.pyfunc.load_model(tire_uri)
    sc = mlflow.pyfunc.load_model(sc_uri)
    policy = PPO.load(policy_path, device="cpu")
    return tire, sc, policy


@st.cache_data(show_spinner="Loading 2024 gold data…")
def _load_gold() -> pd.DataFrame:
    return load_gold_seasons([2024])


@st.cache_resource(show_spinner=False)
def _openf1_client() -> OpenF1Client:
    return OpenF1Client(ttl=25)


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------


def _run_replay(
    race_df: pd.DataFrame,
    driver_num: int | str,
    tire_model: Any,
    sc_model: Any,
    policy: PPO,
) -> pd.DataFrame:
    """Replay the race using actual 2024 team decisions; show policy recommendations.

    The env follows what the team actually did (pit laps, compounds) so that
    positions are realistic.  The policy observes each lap's state and gives
    its recommendation as commentary — this is what Pitwall AI would have said
    had it been in the pit wall that day.
    """
    env = F1RaceEnv(race_df, driver_num, tire_model, sc_model)
    obs, _ = env.reset()

    agent_gold = race_df[race_df["driver_number"] == driver_num].sort_values("lap_number")

    rows: list[dict[str, Any]] = []
    terminated = False

    while not terminated:
        current_lap = env._lap

        # Policy recommendation for this state (commentary only — not executed)
        action, label = recommend_action(obs, policy)
        probs = action_probabilities(obs, policy)

        # --- Rule-based overrides on top of the policy ---

        # 1. Wet-weather override: policy only knows dry compounds (SOFT/MEDIUM/HARD).
        #    Switch to INTER when >50% of field is on wet compounds; WET when >80%.
        wet_frac = (
            float(env._wet_fraction_arr[current_lap]) if current_lap <= env._total_laps else 0.0
        )
        on_wet = env._compound in (3, 4)
        if not on_wet:
            if wet_frac >= 0.8:
                action, label = 5, "Pit — WET"
            elif wet_frac >= 0.5:
                action, label = 4, "Pit — INTER"
        elif on_wet and wet_frac < 0.2:
            action, label = 2, "Pit — MEDIUM"  # conditions dried, switch back

        # 2. Cliff override: nudge policy to pit when past the compound's optimal window.
        cliff = env._cliff_lap.get(env._compound, 35)
        laps_left = env._total_laps - current_lap
        if action == 0 and env._tyre_life >= cliff + 3 and laps_left > 5:
            best = {0: 1, 1: 2, 2: 3}.get(env._compound, 2)  # SOFT→MEDIUM, else HARD
            action, label = best, f"Pit — {['SOFT', 'MEDIUM', 'HARD'][best - 1]} ⚠"

        # Actual team decision drives the simulation
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

        # Translate actual pit decision to action integer for env.step()
        if actual_pit:
            compound_to_action = {0: 1, 1: 2, 2: 3}
            team_action = compound_to_action.get(actual_compound, 1)
        else:
            team_action = 0

        obs, reward, terminated, _, info = env.step(team_action)

        rows.append(
            {
                "lap": info["lap"],
                "position": actual_position if actual_position is not None else info["position"],
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
                "pitted_this_lap": actual_pit,  # what team actually did
                "prob_stay": probs["Stay out"],
                "prob_soft": probs["Pit — SOFT"],
                "prob_medium": probs["Pit — MEDIUM"],
                "prob_hard": probs["Pit — HARD"],
                "actual_pit": actual_pit,
                "actual_compound": actual_compound,
                "reward": reward,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------


def _position_chart(replay: pd.DataFrame, total_laps: int) -> go.Figure:
    """Dual-line position chart: Pitwall AI strategy vs actual team strategy."""
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

    # Simulated position (following actual team decisions)
    fig.add_trace(
        go.Scatter(
            x=replay["lap"],
            y=replay["position"],
            mode="lines+markers",
            name="Simulated position",
            line={"color": "#E8002D", "width": 2},
            marker={"size": 4},
        )
    )

    # Actual position from gold data
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

    # Pit stop markers
    pits = replay[replay["pitted_this_lap"]]
    fig.add_trace(
        go.Scatter(
            x=pits["lap"],
            y=pits["position"],
            mode="markers",
            name="Pit stop",
            marker={"symbol": "triangle-down", "size": 12, "color": "#E8002D"},
            showlegend=True,
        )
    )

    # Laps where AI disagreed with team
    disagreed = replay[
        ((replay["recommended_action"] > 0) & ~replay["actual_pit"])
        | ((replay["recommended_action"] == 0) & replay["actual_pit"])
    ]
    fig.add_trace(
        go.Scatter(
            x=disagreed["lap"],
            y=disagreed["position"],
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


def _render_live_tab(policy: Any) -> None:
    """Render the Live Race tab — OpenF1 real-time strategy recommendations."""
    client = _openf1_client()

    # ── Session picker ───────────────────────────────────────────────────────
    col_sess, col_refresh = st.columns([4, 1])
    with col_sess:
        use_latest = st.checkbox("Track latest session", value=True)
    with col_refresh:
        auto_refresh = st.toggle("Auto-refresh (30 s)", value=False)

    if use_latest:
        try:
            session_meta = client.get_latest_session()
        except Exception as exc:
            st.error(f"OpenF1 API error: {exc}")
            return
        if session_meta is None:
            st.warning("No session found. OpenF1 may not have data for the current event yet.")
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

    gp_name = session_meta.get("meeting_name", "Unknown GP")
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
        st.info("No drivers listed for this session yet.")
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
    action, label = recommend_action(obs, policy)
    probs = action_probabilities(obs, policy)

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
    st.caption(f"Last updated: {ts}  ·  Session key: {session_key}")

    if auto_refresh:
        time.sleep(30)
        st.rerun()
    else:
        if st.button("🔄 Refresh now"):
            st.rerun()


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

        with st.expander("Model details", expanded=False):
            st.caption(selected_model_meta.get("description", ""))
            tags = selected_model_meta.get("tags", {})
            for tag_k, tag_v in tags.items():
                st.caption(f"**{tag_k}:** {tag_v}")

    # ── Load the selected model set (cached per unique paths) ────────────────
    tire_model, sc_model, policy = _load_models(
        model_key=selected_model_key,
        tire_uri=selected_model_meta["tire_uri"],
        sc_uri=selected_model_meta["sc_uri"],
        policy_path=selected_model_meta["policy_path"],
    )

    tab_replay, tab_live = st.tabs(["📊 Race Replay", "🔴 Live Race"])

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1 — Race Replay
    # ═════════════════════════════════════════════════════════════════════════
    with tab_replay:
        season = 2024
        round_options = {f"Round {r} — {n}": r for r, n in ROUND_NAMES.items()}
        rnd_col, _ = st.columns([3, 1])
        with rnd_col:
            selected_label = st.selectbox(
                "Race", list(round_options.keys()), index=9, key="replay_round"
            )
        selected_round = round_options[selected_label]

        gold = _load_gold()
        race_df = gold[(gold["session"] == "R") & (gold["round_number"] == selected_round)].copy()

        if race_df.empty:
            st.error(
                f"No race data found for Round {selected_round}. "
                "Run `make ingest-season SEASON=2024` first."
            )
        else:
            # Driver selector
            driver_options: dict[str, int | str] = {}
            for drv, grp in race_df.groupby("driver_number"):
                abbr = (
                    grp["driver_abbreviation"].iloc[0]
                    if "driver_abbreviation" in grp.columns
                    and pd.notna(grp["driver_abbreviation"].iloc[0])
                    else str(drv)
                )
                team = (
                    grp["team_name"].iloc[0]
                    if "team_name" in grp.columns and pd.notna(grp["team_name"].iloc[0])
                    else ""
                )
                lbl = f"#{drv}  {abbr}" + (f"  · {team}" if team else "")
                driver_options[lbl] = drv  # type: ignore[assignment]

            # Driver picker + run button inline in the tab (not sidebar) so
            # the user sees the output appear directly below the controls.
            drv_col, btn_col = st.columns([4, 1])
            with drv_col:
                selected_driver_label = st.selectbox("Driver", list(driver_options.keys()))
                selected_driver = driver_options[selected_driver_label]
            with btn_col:
                st.write("")  # vertical alignment nudge
                run_btn = st.button("▶  Run", type="primary", use_container_width=True)

            # ── Run replay ────────────────────────────────────────────────────
            cache_key = f"{selected_model_key}_{selected_round}_{selected_driver}"
            replay_stale = (
                "replay_key" not in st.session_state or st.session_state.replay_key != cache_key
            )
            if run_btn or replay_stale:
                with st.spinner("Simulating race…"):
                    st.session_state.replay = _run_replay(
                        race_df, selected_driver, tire_model, sc_model, policy
                    )
                    st.session_state.replay_key = cache_key

            replay: pd.DataFrame = st.session_state.get("replay", pd.DataFrame())
            if replay.empty:
                st.info("Select a race and driver, then click ▶ Run Replay.")
            else:
                total_laps = int(race_df["lap_number"].max())
                event_name = ROUND_NAMES.get(selected_round, f"Round {selected_round}")

                # ── Header ────────────────────────────────────────────────────
                parts = selected_driver_label.split()
                driver_abbr = parts[1] if len(parts) > 1 else str(selected_driver)
                st.markdown(f"## {event_name}  ·  {season}  ·  #{selected_driver} {driver_abbr}")

                # Summary metrics
                final = replay.iloc[-1]
                ai_pos = int(final["position"])
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
                    c2.metric("Position", f"P{int(row['position'])}")

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
                            "position",
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

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2 — Live Race
    # ═════════════════════════════════════════════════════════════════════════
    with tab_live:
        _render_live_tab(policy)


if __name__ == "__main__":
    main()
