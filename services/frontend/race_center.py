"""Streamlit bridge for an animated, recorded-data race broadcast workspace."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from services.live.race_data import (
    RaceFeed,
    RaceFeedError,
    load_live,
    load_replay,
    load_telemetry,
    session_catalog,
    timestamp,
)
from services.simulator.circuits import LOCATION_REFERENCE
from services.simulator.pit_time import load_pit_time_model, predict_pit_time

_component = components.declare_component(
    "pitwall_race_center", path=str(Path(__file__).parent / "race_center_component")
)


@st.cache_resource
def _feed() -> RaceFeed:
    return RaceFeed()


@st.cache_data(show_spinner=False)
def _replay(meta: dict[str, Any]) -> dict[str, Any]:
    return load_replay(_feed(), meta)


@st.cache_data(show_spinner=False)
def _telemetry(key: int, driver: int, origin: str, duration: float) -> list[list[Any]]:
    return load_telemetry(_feed(), key, driver, origin, duration)


def _handle(event: Any) -> None:
    if not isinstance(event, dict) or st.session_state.get("rc_event") == event.get("nonce"):
        return
    st.session_state.rc_event = event.get("nonce")
    action, value = event.get("action"), event.get("value", "")
    if action == "driver":
        st.session_state.rc_driver = int(value)
    elif action == "session":
        st.session_state.rc_session = int(value)
    elif action == "year":
        st.session_state.rc_year = int(value)
        st.session_state.pop("rc_session", None)
    elif action == "mode" and value in ("replay", "live"):
        st.session_state.rc_mode = value
    elif action == "lab":
        st.session_state.workspace = "lab"
    else:
        return
    st.rerun()


def render_race_center() -> None:
    """Render a single client-side timeline; Python only reloads on data changes."""
    st.markdown(
        """<style>
        [data-testid="stHeader"], [data-testid="stSidebar"], #MainMenu, footer {display:none}
        .block-container {max-width:none; padding:0.7rem 1rem 0!important}
        [data-testid="stAppViewContainer"] {background:#0a0c10}
        [data-testid="stElementContainer"]:has(iframe) {line-height:0}
        </style>""",
        unsafe_allow_html=True,
    )
    feed = _feed()
    year = int(st.session_state.get("rc_year", datetime.now().year))
    driver = int(st.session_state.get("rc_driver", 1))
    mode = st.session_state.get("rc_mode", "replay")
    try:
        catalog = session_catalog(feed, year)
    except RaceFeedError as error:
        st.error(f"Race archive unavailable. {error}")
        if st.button("Retry archive"):
            st.rerun()
        return
    meta = next(
        (item for item in catalog if item["session_key"] == st.session_state.get("rc_session")),
        next(
            (item for item in catalog if item["circuit_short_name"] == "Monza"),
            catalog[-1] if catalog else None,
        ),
    )
    base: dict[str, Any] = {
        "year": year,
        "years": list(range(datetime.now().year, 2022, -1)),
        "catalog": catalog,
        "mode": mode,
        "driver": driver,
    }

    def pit_estimate(race_meta: dict[str, Any]) -> dict[str, Any] | None:
        """Serve only past-race model statistics; current stops are blended in the UI."""
        try:
            model = load_pit_time_model()
            venue = LOCATION_REFERENCE.get(str(race_meta.get("circuit_short_name", "")), 0)
            return predict_pit_time(model, venue)
        except (OSError, ValueError, KeyError):
            return None

    @st.fragment(run_every="15s" if mode == "live" else None)
    def screen() -> None:
        payload = dict(base)
        try:
            if mode == "live":
                sessions = feed.get("sessions", {"session_key": "latest"}, ttl=15)
                active = next(
                    (
                        item
                        for item in sessions
                        if timestamp(item["date_start"])
                        <= timestamp(datetime.now().astimezone())
                        <= timestamp(item["date_end"])
                    ),
                    None,
                )
                if active is None:
                    payload["notice"] = {
                        "title": "The paddock is quiet.",
                        "detail": (
                            "There is no active session in the OpenF1 feed. Choose a recorded race "
                            "to watch the timing tower and cars in motion."
                        ),
                    }
                elif not feed.authenticated:
                    payload["notice"] = {
                        "title": "Live feed needs a connection.",
                        "detail": (
                            "OpenF1 requires a subscription for real-time data. Configure "
                            "OPENF1_TOKEN or account credentials on the server, then restart. "
                            "Recorded races are ready to watch."
                        ),
                    }
                else:
                    payload["race"], payload["telemetry"], payload["freshness"] = load_live(
                        feed, active, driver
                    )
                    payload["pit_time"] = pit_estimate(active)
            elif meta:
                with st.spinner(
                    "Preparing recorded positions and timing… First load caches the race locally."
                ):
                    payload["race"] = _replay(meta)
                    payload["pit_time"] = pit_estimate(meta)
                    payload["telemetry"] = _telemetry(
                        int(meta["session_key"]),
                        driver,
                        payload["race"]["origin"],
                        payload["race"]["duration"],
                    )
            else:
                payload["notice"] = {
                    "title": "No completed races yet.",
                    "detail": "Choose another season to explore the archive.",
                }
        except (RaceFeedError, ValueError, OSError) as error:
            payload["notice"] = {
                "title": "The feed is taking a pit stop.",
                "detail": f"{error}. Switch to a recorded race or retry the connection.",
            }
        event = _component(payload=payload, key="race_center", default=None)
        _handle(event)

    screen()
