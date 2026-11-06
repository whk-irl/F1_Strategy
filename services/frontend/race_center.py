"""Streamlit bridge for an animated, recorded-data race broadcast workspace."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from services.live.free_timing import load_free_snapshot
from services.live.race_data import (
    RaceFeed,
    RaceFeedError,
    RaceFeedPendingError,
    load_free_live,
    load_live,
    load_replay,
    load_telemetry,
    record_live_decision,
    select_live_session,
    session_catalog,
    timestamp,
)
from services.live.weekend import load_weekend_summary
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


ADVISOR_ENGINES = ("advisor", "ensemble")
_ADVISOR_RECORDS: dict[tuple[int, str], dict[str, Any]] = {}
_ADVISOR_CACHE_RACES = 8


@st.cache_resource(show_spinner=False)
def _strategy_advisor(engine: str) -> Any:
    """The default planner engine never imports PyTorch or loads ensemble weights."""
    import json

    from ml.strategy_serving.advisor import SERVING_CONFIG, StrategyAdvisor

    if engine == "ensemble":
        return StrategyAdvisor.from_config()
    return StrategyAdvisor(Path(json.loads(SERVING_CONFIG.read_text())["priors"]))


def _load_advisor_strategy(race: dict[str, Any], engine: str) -> dict[str, Any]:
    """Per-lap advisor calls for a recorded race (docs/model-cards/strategy-advisor-v1.md)."""
    from services.live.advisor_strategy import advisor_records

    key = (int(race["id"]), engine)
    if key not in _ADVISOR_RECORDS:
        if len(_ADVISOR_RECORDS) >= _ADVISOR_CACHE_RACES:
            _ADVISOR_RECORDS.pop(next(iter(_ADVISOR_RECORDS)))
        _ADVISOR_RECORDS[key] = advisor_records(race, _strategy_advisor(engine), engine)  # type: ignore[arg-type]
    return _ADVISOR_RECORDS[key]


def _attach_strategy_payload(payload: dict[str, Any]) -> None:
    """Attach advisor calls for recorded races; an unavailable engine never falls back."""
    engine = st.session_state.get("rc_strategy_engine", "reference")
    if engine not in ADVISOR_ENGINES:
        payload["strategy_engine"] = "reference"
        return
    model = None
    if payload.get("mode") != "replay":
        payload["strategy_model_error"] = (
            "The strategy advisor runs on recorded races only; live use is not yet validated."
        )
    elif payload.get("race"):
        try:
            model = _load_advisor_strategy(payload["race"], engine)
        except Exception as error:  # the UI shows a held call, never a stack trace
            payload["strategy_model_error"] = "The strategy advisor is unavailable."
            payload["strategy_model_error_code"] = type(error).__name__
    payload["strategy_model"] = model
    payload["strategy_available"] = model is not None
    payload["strategy_engine"] = engine


def _handle(event: Any) -> None:
    if not isinstance(event, dict) or st.session_state.get("rc_event") == event.get("nonce"):
        return
    st.session_state.rc_event = event.get("nonce")
    action, value = event.get("action"), event.get("value", "")
    if action == "strategy_record":
        record_live_decision(_feed(), value, load_weekend_summary())
        return
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
    elif action == "strategy_engine" and value in ("reference", *ADVISOR_ENGINES):
        st.session_state.rc_strategy_engine = value
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
    default_mode = "live" if os.getenv("PITWALL_START_LIVE") == "1" else "replay"
    mode = st.session_state.get("rc_mode", default_mode)
    provider = os.getenv("PITWALL_LIVE_PROVIDER", "auto").lower()
    try:
        catalog = session_catalog(feed, year) if mode != "live" else []
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
                active = select_live_session(feed, load_weekend_summary())
                context = load_weekend_summary(active) if active else None
                payload["weekend"] = context
                if active is None:
                    payload["notice"] = {
                        "title": "The paddock is quiet.",
                        "detail": (
                            "There is no active session in the OpenF1 feed. Choose a recorded race "
                            "to watch the timing tower and cars in motion."
                        ),
                    }
                elif timestamp(active["date_start"]) > timestamp(datetime.now().astimezone()):
                    completed = ", ".join(
                        item["code"] for item in (context or {}).get("sessions", [])
                    )
                    top_three = ", ".join(
                        f"P{row.get('position', idx + 1)} "
                        f"{row.get('name_acronym', row['driver_number'])}"
                        for idx, row in enumerate((context or {}).get("qualifying", [])[:3])
                    )
                    start_label = timestamp(active["date_start"]).tz_convert("America/Chicago")
                    payload["notice"] = {
                        "title": "United States GP · race preparation",
                        "detail": (
                            "Start: "
                            f"{start_label:%a %b %d, %H:%M %Z} · "
                            f"{(context or {}).get('scheduled_laps', 'Unconfirmed')} laps. "
                            f"Updated: {completed or 'weekend data pending'}. "
                            f"Qualifying: {top_three or 'classification pending'}. "
                            f"{(context or {}).get('grid_status', 'Starting grid pending')}. "
                            "The local free timing collector will be checked at lights out; "
                            "its availability is experimental."
                        ),
                    }
                else:
                    free = load_free_snapshot(expected_meta=active, max_age_s=None)
                    if free is not None and (provider == "free" or not feed.authenticated):
                        result = load_free_live(feed, free, active, driver, context)
                        payload["race"], payload["telemetry"], payload["freshness"] = result
                    elif feed.authenticated and provider != "free":
                        payload["race"], payload["telemetry"], payload["freshness"] = load_live(
                            feed, active, driver, context
                        )
                    else:
                        payload["notice"] = {
                            "title": "Waiting for the local timing collector.",
                            "detail": (
                                "No matching free timing snapshot has arrived. Start the local "
                                "FastF1 public SignalR collector and keep it running alongside "
                                "Streamlit. Live availability must be verified during the session. "
                                "The alternative OpenF1 live feed requires a subscription."
                            ),
                        }
                    payload["pit_time"] = pit_estimate(active)
            elif meta:
                with st.spinner(
                    "Preparing recorded positions and timing… First load caches the race locally."
                ):
                    payload["race"] = _replay(meta)
                    payload["weekend"] = load_weekend_summary(meta)
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
        except RaceFeedPendingError as error:
            payload["notice"] = {
                "title": "Waiting for the first race samples.",
                "detail": str(error),
            }
        except (RaceFeedError, ValueError, OSError) as error:
            payload["notice"] = {
                "title": "The feed is taking a pit stop.",
                "detail": f"{error}. Switch to a recorded race or retry the connection.",
            }
        _attach_strategy_payload(payload)
        event = _component(payload=payload, key="race_center", default=None)
        _handle(event)

    screen()
    if mode == "live" and (prepared := load_weekend_summary()):
        with st.expander("United States GP · practice, qualifying and starting grid"):
            st.caption(
                f"Updated {prepared['updated_at']} · {prepared.get('rows', 0):,} measured laps · "
                f"{prepared.get('grid_status', 'Starting grid pending')}. "
                "These sessions inform the nominal tyre estimate; they do not retrain the policy."
            )
            sessions = prepared.get("sessions", [])
            tabs = st.tabs([item["code"] for item in sessions] + ["Starting grid", "Tyre evidence"])
            for tab, item in zip(tabs, sessions, strict=False):
                with tab:
                    rows = []
                    for result in item.get("classification", []):
                        duration = result.get("duration")
                        rows.append(
                            {
                                "Position": result.get("position"),
                                "Driver": result.get("name_acronym"),
                                "Team": result.get("team_name"),
                                "Laps": result.get("number_of_laps"),
                                "Time (s)": duration if not isinstance(duration, list) else None,
                                **(
                                    {f"Q{idx + 1} (s)": value for idx, value in enumerate(duration)}
                                    if isinstance(duration, list)
                                    else {}
                                ),
                            }
                        )
                    st.dataframe(rows, hide_index=True, use_container_width=True)
            with tabs[-2]:
                st.dataframe(
                    [
                        {
                            "Grid": row.get("position"),
                            "Car": row.get("driver_number"),
                            "Driver": row.get("name_acronym"),
                            "Team": row.get("team_name"),
                            "Qualifying lap (s)": row.get("duration"),
                        }
                        for row in prepared.get("starting_grid", [])
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
            with tabs[-1]:
                calibration = prepared.get("calibration", {})
                st.dataframe(
                    [
                        {
                            "Compound": compound,
                            "Clean laps": evidence.get("clean_laps"),
                            "Long stints": evidence.get("usable_stints"),
                            "Observed pace slope (s/lap)": evidence.get("observed_slope_s_per_lap"),
                            "Source": evidence.get("source"),
                        }
                        for compound, evidence in calibration.get("compound_evidence", {}).items()
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(calibration.get("note", "Practice calibration pending."))
