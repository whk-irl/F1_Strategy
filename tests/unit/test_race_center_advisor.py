"""Race Center advisor engines: bridge, race-bundle conversion and client contract."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
from ml.strategy.race_model import TyreParams
from ml.strategy_serving.advisor import StrategyAdvisor
from ml.strategy_serving.priors import GLOBAL
from services.frontend import race_center as bridge
from services.live import advisor_strategy as adv_rc


@pytest.fixture
def bridge_state(monkeypatch):
    monkeypatch.setattr(bridge.st, "session_state", {})
    return bridge.st.session_state


def test_advisor_engine_attaches_replay_records(bridge_state, monkeypatch):
    bridge_state["rc_strategy_engine"] = "advisor"
    monkeypatch.setattr(
        bridge, "_load_advisor_strategy", lambda race, engine: {"records": [{"engine": engine}]}
    )
    payload = {"race": {"id": 11396}, "mode": "replay"}
    bridge._attach_strategy_payload(payload)
    assert payload["strategy_engine"] == "advisor"
    assert payload["strategy_available"] is True
    assert payload["strategy_model"]["records"][0]["engine"] == "advisor"


def test_reference_engine_loads_no_advisor(bridge_state, monkeypatch):
    monkeypatch.setattr(bridge, "_load_advisor_strategy", lambda *a: pytest.fail("not selected"))
    payload = {"race": {"id": 11396}, "mode": "replay"}
    bridge._attach_strategy_payload(payload)
    assert payload == {"race": {"id": 11396}, "mode": "replay", "strategy_engine": "reference"}


def test_advisor_engine_is_replay_only_and_never_falls_back(bridge_state, monkeypatch):
    bridge_state["rc_strategy_engine"] = "ensemble"
    monkeypatch.setattr(bridge, "_load_advisor_strategy", lambda *a: pytest.fail("no live advisor"))
    payload = {"race": {"id": 11396}, "mode": "live"}
    bridge._attach_strategy_payload(payload)
    assert payload["strategy_engine"] == "ensemble"
    assert payload["strategy_available"] is False
    assert "recorded races only" in payload["strategy_model_error"]


def test_advisor_failure_is_reported_without_details(bridge_state, monkeypatch):
    bridge_state["rc_strategy_engine"] = "advisor"

    def fail(*a):
        raise RuntimeError("secret path details")

    monkeypatch.setattr(bridge, "_load_advisor_strategy", fail)
    payload = {"race": {"id": 11396}, "mode": "replay"}
    bridge._attach_strategy_payload(payload)
    assert payload["strategy_engine"] == "advisor"
    assert payload["strategy_model"] is None
    assert payload["strategy_model_error_code"] == "RuntimeError"
    assert "secret path details" not in json.dumps(payload)


class _SessionState(dict):
    """Streamlit session state supports both item and attribute access."""

    __getattr__ = dict.get

    def __setattr__(self, key, value):
        self[key] = value


@pytest.mark.parametrize("engine", ["reference", "advisor", "ensemble"])
def test_engine_selection_event_is_accepted(monkeypatch, engine):
    state = _SessionState()
    monkeypatch.setattr(bridge.st, "session_state", state)
    monkeypatch.setattr(bridge.st, "rerun", lambda: None)
    bridge._handle({"nonce": 1, "action": "strategy_engine", "value": engine})
    assert state["rc_strategy_engine"] == engine


def test_unknown_engine_event_is_ignored(monkeypatch):
    state = _SessionState()
    monkeypatch.setattr(bridge.st, "session_state", state)
    monkeypatch.setattr(bridge.st, "rerun", lambda: pytest.fail("no rerun"))
    bridge._handle({"nonce": 2, "action": "strategy_engine", "value": "candidate"})
    assert "rc_strategy_engine" not in state


def test_neutral_intervals_from_race_control():
    msgs = [
        {"t": 100.0, "category": "SafetyCar", "message": "VSC DEPLOYED"},
        {"t": 150.0, "category": "SafetyCar", "message": "VSC ENDING"},
        {"t": 400.0, "category": "SafetyCar", "message": "SAFETY CAR DEPLOYED"},
        {"t": 700.0, "category": "SafetyCar", "message": "SAFETY CAR IN THIS LAP"},
        {"t": 900.0, "category": "Flag", "message": "TRACK CLEAR"},
    ]
    assert adv_rc.neutral_intervals(msgs, 90.0) == [(100.0, 165.0, "VSC"), (400.0, 790.0, "SC")]


def _bundle(n: int = 30) -> dict:
    laps, stints, pits = {}, [], []
    for d in range(1, 9):
        rows, t = [], 0.0
        for lap in range(1, n + 1):
            stop = lap == 12 + d
            age = lap if lap <= 12 + d else lap - (12 + d)
            dur = 90.0 + 0.08 * age + (20.0 if stop or lap == 13 + d else 0.0) + 0.05 * d
            rows.append([t, t + dur, lap, dur, None, None, None])
            t += dur
        laps[str(d)] = rows
        stints += [
            {
                "driver_number": d,
                "lap_start": 1,
                "lap_end": 12 + d,
                "compound": "MEDIUM",
                "tyre_age_at_start": 0,
            },
            {
                "driver_number": d,
                "lap_start": 13 + d,
                "lap_end": n,
                "compound": "HARD",
                "tyre_age_at_start": 0,
            },
        ]
        pits.append({"driver_number": d, "lap_number": 12 + d})
    return {
        "id": 1,
        "meta": {"circuit_short_name": "Monza"},
        "total_laps": n,
        "laps": laps,
        "stints": stints,
        "pits": pits,
        "messages": [],
    }


def _advisor(tmp_path: Path) -> StrategyAdvisor:
    params = TyreParams(deg=(0.12, 0.08, 0.05), p_neutral=0.05)
    path = tmp_path / "priors.json"
    path.write_text(
        json.dumps(
            {"format_version": 1, "priors": {GLOBAL: asdict(params), "monza": asdict(params)}}
        )
    )
    return StrategyAdvisor(path)


def test_records_follow_the_client_timeline_contract(tmp_path):
    out = adv_rc.advisor_records(_bundle(), _advisor(tmp_path), "advisor")
    recs = out["records"]
    assert out["coverage"] == {"drivers": 8, "rows": 240, "circuit": "monza"}
    assert out["promoted"] is True
    for r in recs:
        assert r["input_cutoff_lap"] == r["completed_lap"] == r["lap"] - 1
        assert r["action"] in ("stay_out", "pit_now", "run_to_flag")
    stays = [r for r in recs if r["action"] == "stay_out"]
    assert all(min(c["cost_s"] for c in r["candidates"]) == 0.0 for r in stays)
    first_stint_pits = [
        r for r in recs if r["action"] == "pit_now" and r["lap"] <= 12 + r["driver"]
    ]
    assert all(r["fit"] != "MEDIUM" for r in first_stint_pits)  # second compound required
    assert all(r["action"] == "run_to_flag" for r in recs if r["lap"] == 30)


def test_client_routes_advisor_engines_to_model_panel():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js unavailable")
    source = (Path(bridge.__file__).parent / "race_center_component" / "app.js").read_text(
        encoding="utf-8"
    )
    consts = source.split("const MODEL_ENGINES=", 1)[1].split("function selectModelRecord(", 1)[0]
    script = (
        "const MODEL_ENGINES="
        + consts
        + r"""
const assert=require('node:assert/strict');
assert.deepEqual(MODEL_ENGINES,['advisor','ensemble']);
assert.ok(ADVISOR_NOTES.advisor.includes('Released'));
assert.ok(ADVISOR_NOTES.ensemble.includes('7e8800ab93b81c68'));
console.log('routing passed');
"""
    )
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "routing passed" in result.stdout
    assert (
        "if(MODEL_ENGINES.includes(state.payload.strategy_engine)){updateModelStrategy();return;}"
        in source
    )
