"""Live shadow: rebuild history from lossy live snapshots, call once per lap, hold after stops."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from ml.strategy.race_model import TyreParams
from ml.strategy_serving.advisor import StrategyAdvisor
from ml.strategy_serving.priors import GLOBAL
from services.live import advisor_shadow as shadow

LAP = 95.0


def _advisor(tmp_path: Path) -> StrategyAdvisor:
    p = tmp_path / "priors.json"
    params = asdict(TyreParams(deg=(0.12, 0.08, 0.05)))
    p.write_text(json.dumps({"format_version": 1, "priors": {GLOBAL: params, "usa": params}}))
    return StrategyAdvisor(p)


def _snap(
    t: float, lap_in_progress: int, stints: list[dict], status: str = "1", total: int | None = 30
) -> dict:
    """Live-format snapshot: last two lap starts only, no durations, latest_completed."""
    start = lambda lap: (lap - 1) * LAP  # noqa: E731
    laps = {
        "7": [
            [start(lap), None, lap, None, None, None, None]
            for lap in (lap_in_progress - 1, lap_in_progress)
            if lap >= 1
        ]
    }
    done = lap_in_progress - 1
    latest = (
        {"7": [start(done), start(done) + LAP, done, LAP, None, None, None]} if done >= 1 else {}
    )
    return {
        "t": t, "meta": {"session_key": 1, "circuit_short_name": "Austin"},
        "distance_confirmed": total is not None, "total_laps": total or 0,
        "track_status": status, "laps": laps, "latest_completed": latest, "stints": stints,
    }  # fmt: skip


def test_history_accumulates_lap_times_and_anchors_unknown_stint_starts():
    h = shadow.LiveFieldHistory()
    for lap in range(1, 25):
        stints = [
            {
                "driver_number": 7,
                "stint_number": 1,
                "compound": "MEDIUM",
                "lap_start": 1,
                "tyre_age_at_start": 0,
            }
        ]
        if lap >= 17:  # stint 2 published with lap_start 0 and its age
            stints.append(
                {
                    "driver_number": 7,
                    "stint_number": 2,
                    "compound": "HARD",
                    "lap_start": 0,
                    "tyre_age": lap - 17,  # laps completed on the set
                }
            )
        h.ingest(_snap((lap - 1) * LAP + 50, lap, stints))
    laps = {f.lap_number: f for f in h.field_laps()}
    assert set(laps) == set(range(1, 24))  # every completed lap, though each snapshot held two
    assert all(f.lap_time_s == pytest.approx(LAP) for f in laps.values())
    assert laps[16].compound == "MEDIUM"
    assert laps[16].pit_in
    assert (laps[17].compound, laps[17].tyre_life_laps, laps[17].pit_out) == ("HARD", 1, True)
    assert laps[23].tyre_life_laps == 7


def test_current_sample_outranks_lagging_lap_start_zero_record():
    d = shadow._Driver(starts={16: 0.0})  # lap rows lag: lap 16 still shown in progress
    shadow.LiveFieldHistory._stint(
        d,
        {
            "driver_number": 7,
            "stint_number": 2,
            "compound": "HARD",
            "lap_start": 22,
            "tyre_age_at_start": 5,
            "current_sample": True,
        },
    )
    shadow.LiveFieldHistory._stint(
        d,
        {"driver_number": 7, "stint_number": 2, "compound": "HARD", "lap_start": 0, "tyre_age": 4},
    )
    assert d.stints[2]["start"] == 17  # from the current sample, not 16 - 4 = 12


def test_one_call_per_car_lap_held_until_stint_feed_catches_up(tmp_path):
    sh = shadow.AdvisorShadow(_advisor(tmp_path))
    stints = [
        {
            "driver_number": 7,
            "stint_number": 1,
            "compound": "MEDIUM",
            "lap_start": 1,
            "tyre_age_at_start": 0,
        }
    ]
    lap_start = 9 * LAP  # lap 10 begins
    assert (
        sh.step(_snap(lap_start + 5, 10, stints)) == []
    )  # too early in the lap: tyres unconfirmed
    calls = sh.step(_snap(lap_start + shadow.STINT_FEED_HOLD_S + 1, 10, stints))
    assert [(c["driver"], c["lap"]) for c in calls] == [(7, 10)]
    assert sh.step(_snap(lap_start + 80, 10, stints)) == []  # never re-issued
    assert calls[0]["field_laps_used"] == 1  # only lap 9 had been published to this worker


def test_no_calls_without_confirmed_distance(tmp_path):
    sh = shadow.AdvisorShadow(_advisor(tmp_path))
    stints = [
        {
            "driver_number": 7,
            "stint_number": 1,
            "compound": "MEDIUM",
            "lap_start": 1,
            "tyre_age_at_start": 0,
        }
    ]
    assert sh.step(_snap(9 * LAP + 60, 10, stints, total=None)) == []


def test_neutral_status_marks_laps():
    h = shadow.LiveFieldHistory()
    stints = [
        {
            "driver_number": 7,
            "stint_number": 1,
            "compound": "MEDIUM",
            "lap_start": 1,
            "tyre_age_at_start": 0,
        }
    ]
    for lap in range(1, 6):
        h.ingest(_snap((lap - 1) * LAP + 50, lap, stints, status="6" if lap == 3 else "1"))
    status = {f.lap_number: f.track_status for f in h.field_laps()}
    assert status[3] == "6"
    assert status[2] == "1"
