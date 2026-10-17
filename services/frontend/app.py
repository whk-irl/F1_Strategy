"""Pitwall race center, with the original model explorer available as Strategy Lab."""

from __future__ import annotations

import sys
from importlib import reload
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv()

_RELEASE = "2026-10-16-measured-pit-time"


def _refresh_release_modules() -> None:
    """Refresh imported modules after Streamlit Cloud updates files in place."""
    if st.session_state.get("_pitwall_release") == _RELEASE:
        return
    for name in (
        "services.live.race_data",
        "services.simulator.circuits",
        "services.simulator.pit_time",
        "services.frontend.race_center",
        "services.frontend.strategy_lab",
    ):
        if name in sys.modules:
            reload(sys.modules[name])
    st.session_state._pitwall_release = _RELEASE


def main() -> None:
    st.set_page_config(page_title="Pitwall • Race Center", page_icon="🏁", layout="wide")
    _refresh_release_modules()
    if st.session_state.get("workspace") == "lab":
        if st.button("← Back to Race Center"):
            st.session_state.workspace = "race"
            st.rerun()
        from services.frontend.strategy_lab import main as strategy_lab

        strategy_lab()
    else:
        from services.frontend.race_center import render_race_center

        render_race_center()


if __name__ == "__main__":
    main()
