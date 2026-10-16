"""Pitwall race center, with the original model explorer available as Strategy Lab."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv()


def main() -> None:
    st.set_page_config(page_title="Pitwall • Race Center", page_icon="🏁", layout="wide")
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
