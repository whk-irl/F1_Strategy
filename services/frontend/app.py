"""Pitwall AI — Streamlit demo app.

2024 race replay mode: pick any round and driver, watch the PPO policy's
lap-by-lap pit recommendations alongside what the team actually did.

Run with:
    streamlit run services/frontend/app.py
"""

from __future__ import annotations

import os

import mlflow.pyfunc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ml.models._loader import load_gold_seasons
from ml.models.strategy_policy.predict import (
    action_probabilities,
    load_policy,
    recommend_action,
)
from services.simulator.env import F1RaceEnv
from services.simulator.lap_simulator import PIT_LOSS_S

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUND_NAMES: dict[int, str] = {
    1: "Bahrain GP", 2: "Saudi Arabian GP", 3: "Australian GP",
    4: "Japanese GP", 5: "Chinese GP", 6: "Miami GP",
    7: "Emilia Romagna GP", 8: "Monaco GP", 9: "Canadian GP",
    10: "Spanish GP", 11: "Austrian GP", 12: "British GP",
    13: "Hungarian GP", 14: "Belgian GP", 15: "Dutch GP",
    16: "Italian GP", 17: "Azerbaijan GP", 18: "Singapore GP",
    19: "United States GP", 20: "Mexico City GP", 21: "São Paulo GP",
    22: "Las Vegas GP", 23: "Qatar GP", 24: "Abu Dhabi GP",
}

COMPOUND_LABEL: dict[int, str] = {
    0: "SOFT", 1: "MEDIUM", 2: "HARD", 3: "INTER", 4: "WET", 5: "UNKNOWN",
}
COMPOUND_COLOR: dict[int, str] = {
    0: "#FF3333", 1: "#FFD700", 2: "#DDDDDD", 3: "#33CC33", 4: "#3399FF", 5: "#888888",
}
ACTION_COLOR: dict[int, str] = {
    0: "#444444", 1: "#FF3333", 2: "#FFD700", 3: "#DDDDDD",
}

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading models…")
def _load_models() -> tuple:
    tracking_uri = os.getenv("PITWALL_MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    tire = mlflow.pyfunc.load_model(
        os.getenv("PITWALL_TIRE_MODEL_URI", "models:/pitwall-tire-degradation/latest")
    )
    sc = mlflow.pyfunc.load_model(
        os.getenv("PITWALL_SC_MODEL_URI", "models:/pitwall-safety-car/latest")
    )
    policy = load_policy()
    return tire, sc, policy


@st.cache_data(show_spinner="Loading 2024 gold data…")
def _load_gold() -> pd.DataFrame:
    return load_gold_seasons([2024])


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------


def _run_replay(
    race_df: pd.DataFrame,
    driver_num: int | str,
    tire_model: object,
    sc_model: object,
    policy: object,
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

    rows: list[dict] = []
    terminated = False

    while not terminated:
        current_lap = env._lap

        # Policy recommendation for this state (commentary only — not executed)
        action, label = recommend_action(obs, policy)
        probs = action_probabilities(obs, policy)

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

        rows.append({
            "lap": info["lap"],
            "position": info["position"],
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
        })

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
            x0=lap - 0.5, x1=lap + 0.5,
            fillcolor="yellow", opacity=0.15, line_width=0,
        )

    # Simulated position (following actual team decisions)
    fig.add_trace(go.Scatter(
        x=replay["lap"], y=replay["position"],
        mode="lines+markers", name="Simulated position",
        line=dict(color="#E8002D", width=2),
        marker=dict(size=4),
    ))

    # Actual position from gold data
    actual = replay.dropna(subset=["actual_position"])
    fig.add_trace(go.Scatter(
        x=actual["lap"], y=actual["actual_position"],
        mode="lines", name="Actual (2024)",
        line=dict(color="#888888", width=1.5, dash="dot"),
    ))

    # Pit stop markers
    pits = replay[replay["pitted_this_lap"]]
    fig.add_trace(go.Scatter(
        x=pits["lap"], y=pits["position"],
        mode="markers", name="Pit stop",
        marker=dict(symbol="triangle-down", size=12, color="#E8002D"),
        showlegend=True,
    ))

    # Laps where AI disagreed with team
    disagreed = replay[
        ((replay["recommended_action"] > 0) & ~replay["actual_pit"]) |
        ((replay["recommended_action"] == 0) & replay["actual_pit"])
    ]
    fig.add_trace(go.Scatter(
        x=disagreed["lap"], y=disagreed["position"],
        mode="markers", name="AI disagreed",
        marker=dict(symbol="x", size=10, color="#FFD700"),
        showlegend=True,
    ))

    fig.update_layout(
        yaxis=dict(autorange="reversed", title="Position", dtick=1),
        xaxis=dict(title="Lap", range=[1, total_laps]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=10, b=40),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,20,30,1)",
        font=dict(color="#FFFFFF"),
        yaxis_gridcolor="#333",
        xaxis_gridcolor="#333",
    )
    return fig


def _prob_bar(label: str, prob: float, color: str) -> None:
    cols = st.columns([3, 7])
    cols[0].caption(label)
    cols[1].progress(prob, text=f"{prob:.1%}")


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
    st.markdown("""
    <style>
        .block-container { padding-top: 1rem; }
        .metric-label { font-size: 0.75rem; color: #999; }
        .compound-badge {
            display: inline-block; padding: 2px 10px;
            border-radius: 12px; font-weight: 700; font-size: 0.85rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🏎️ Pitwall AI")
        st.caption("F1 Race Strategy Engine · 2024 Season Replay")
        st.divider()

        season = 2024
        round_options = {f"Round {r} — {n}": r for r, n in ROUND_NAMES.items()}
        selected_label = st.selectbox("Race", list(round_options.keys()), index=9)  # Spain default — highest tyre wear
        selected_round = round_options[selected_label]

        st.divider()
        st.caption("Model info")
        st.caption("PPO · 20-feature obs · 128×64 MLP")
        st.caption("Trained on 2024 gold data (all 24 rounds)")

    # ── Load resources ───────────────────────────────────────────────────────
    tire_model, sc_model, policy = _load_models()
    gold = _load_gold()

    race_df = gold[
        (gold["session"] == "R") & (gold["round_number"] == selected_round)
    ].copy()

    if race_df.empty:
        st.error(f"No race data found for Round {selected_round}. Run `make ingest-season SEASON=2024` first.")
        return

    # Driver selector
    driver_options: dict[str, int | str] = {}
    for drv, grp in race_df.groupby("driver_number"):
        abbr = (
            grp["driver_abbreviation"].iloc[0]
            if "driver_abbreviation" in grp.columns and pd.notna(grp["driver_abbreviation"].iloc[0])
            else str(drv)
        )
        team = (
            grp["team_name"].iloc[0]
            if "team_name" in grp.columns and pd.notna(grp["team_name"].iloc[0])
            else ""
        )
        label = f"#{drv}  {abbr}" + (f"  · {team}" if team else "")
        driver_options[label] = drv

    with st.sidebar:
        selected_driver_label = st.selectbox("Driver", list(driver_options.keys()))
        selected_driver = driver_options[selected_driver_label]
        run_btn = st.button("▶  Run Replay", type="primary", use_container_width=True)

    # ── Run replay ───────────────────────────────────────────────────────────
    cache_key = f"{selected_round}_{selected_driver}"
    if run_btn or ("replay_key" not in st.session_state) or (st.session_state.replay_key != cache_key):
        with st.spinner("Simulating race…"):
            st.session_state.replay = _run_replay(
                race_df, selected_driver, tire_model, sc_model, policy
            )
            st.session_state.replay_key = cache_key

    replay: pd.DataFrame = st.session_state.get("replay", pd.DataFrame())
    if replay.empty:
        st.info("Select a race and driver, then click ▶ Run Replay.")
        return

    total_laps = int(race_df["lap_number"].max())
    event_name = ROUND_NAMES.get(selected_round, f"Round {selected_round}")

    # ── Header ───────────────────────────────────────────────────────────────
    driver_abbr = selected_driver_label.split()[1] if len(selected_driver_label.split()) > 1 else str(selected_driver)
    st.markdown(f"## {event_name}  ·  {season}  ·  #{selected_driver} {driver_abbr}")

    # Summary metrics
    final = replay.iloc[-1]
    ai_pos = int(final["position"])
    actual_pos = int(final["actual_position"]) if pd.notna(final["actual_position"]) else "—"
    n_pits = int(replay["pitted_this_lap"].sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pitwall AI finish", f"P{ai_pos}")
    m2.metric("Actual finish", f"P{actual_pos}")
    m3.metric("AI pit stops", n_pits)
    m4.metric("Total laps", total_laps)

    # ── Position chart ───────────────────────────────────────────────────────
    st.plotly_chart(_position_chart(replay, total_laps), use_container_width=True)
    st.caption("🟡 Yellow bands = Safety Car / VSC period  ·  ▼ = Pit stop")

    st.divider()

    # ── Lap explorer ─────────────────────────────────────────────────────────
    st.subheader("Lap-by-lap explorer")
    selected_lap = st.slider("Lap", 1, len(replay), 1)
    row = replay[replay["lap"] == selected_lap]
    if row.empty:
        row = replay.iloc[selected_lap - 1: selected_lap]
    row = row.iloc[0]

    left, right = st.columns(2)

    # Left — race state
    with left:
        st.markdown("#### Race state")
        compound = int(row["compound"])
        comp_color = COMPOUND_COLOR.get(compound, "#888")
        comp_name = COMPOUND_LABEL.get(compound, "?")
        st.markdown(
            f'<span class="compound-badge" style="background:{comp_color};'
            f'color:{"#111" if compound in (1, 2, 3) else "#fff"}">'
            f'{comp_name}</span>',
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

    # Right — policy recommendation
    with right:
        st.markdown("#### Pitwall AI recommendation")
        action = int(row["recommended_action"])
        rec_color = ACTION_COLOR.get(action, "#888")
        st.markdown(
            f'<h3 style="color:{rec_color}">{row["recommended_label"]}</h3>',
            unsafe_allow_html=True,
        )

        _prob_bar("Stay out",    row["prob_stay"],   "#444")
        _prob_bar("Pit — SOFT",  row["prob_soft"],   "#FF3333")
        _prob_bar("Pit — MEDIUM",row["prob_medium"], "#FFD700")
        _prob_bar("Pit — HARD",  row["prob_hard"],   "#DDDDDD")

        total_pit_prob = row["prob_soft"] + row["prob_medium"] + row["prob_hard"]
        if action == 0 and total_pit_prob > 0.25:
            st.warning(f"Pit window opening — {total_pit_prob:.0%} chance pit is optimal")

        st.divider()
        actual_decision = "Pit" if row["actual_pit"] else "Stay out"
        agreed = (action == 0 and not row["actual_pit"]) or (action > 0 and row["actual_pit"])
        icon = "✅" if agreed else "⚠️"
        st.markdown(f"**Actual team decision:** {actual_decision}  {icon}")
        if not agreed:
            if action > 0 and not row["actual_pit"]:
                st.caption("Pitwall AI would pit here — team stayed out.")
            else:
                st.caption("Team pitted — Pitwall AI would stay out.")

    # ── Stint breakdown table ─────────────────────────────────────────────────
    with st.expander("Full race log"):
        display = replay[[
            "lap", "position", "actual_position",
            "compound", "tyre_life", "sim_lap_time_s",
            "gap_ahead_s", "gap_behind_s",
            "recommended_label", "recommended_action", "actual_pit",
        ]].copy()
        display["compound"] = display["compound"].map(COMPOUND_LABEL)
        display["gap_ahead_s"] = display["gap_ahead_s"].apply(
            lambda x: f"{x:.1f}s" if x < 59 else "—"
        )
        display["gap_behind_s"] = display["gap_behind_s"].apply(
            lambda x: f"{x:.1f}s" if x < 59 else "—"
        )
        # AI would pit = recommended_action > 0
        display["ai_would_pit"] = display["recommended_action"] > 0
        display.drop(columns=["recommended_action"], inplace=True)
        display.columns = [
            "Lap", "AI Pos", "Actual Pos",
            "Compound", "Tyre Age", "Sim Lap (s)",
            "Gap Ahead", "Gap Behind",
            "AI Recommendation", "Team Pitted", "AI Would Pit",
        ]
        st.dataframe(display, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
