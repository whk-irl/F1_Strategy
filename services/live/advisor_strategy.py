"""Race Center bridge for the strategy advisor (docs/model-cards/strategy-advisor-v1.md).

Turns a recorded Race Center bundle (OpenF1 laps, stints, pits, race control) into
per-driver, per-lap advisor calls in the component's model-record format, so the
existing model panel can display them. Replay only: the advisor has not been
validated on live feeds.

Engines: ``advisor`` = released DP planner; ``ensemble`` = frozen opt-in candidate.
A call for lap L (box at the end of lap L) uses field laps through L-1, matching
the evaluated protocol; laps are indexed by lap number, not wall-clock time.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from ml.strategy.planner import Plan, solve
from ml.strategy.race_model import TyreParams
from ml.strategy_serving.advisor import COMPOUNDS, CarState, FieldLap, StrategyAdvisor

# OpenF1 circuit_short_name -> strategy circuit key (race_model.CALENDAR names).
CIRCUITS = {
    "Sakhir": "bahrain", "Jeddah": "saudi", "Melbourne": "australia", "Suzuka": "japan",
    "Shanghai": "china", "Miami": "miami", "Imola": "imola", "Monte Carlo": "monaco",
    "Montreal": "canada", "Catalunya": "spain", "Spielberg": "austria", "Silverstone": "britain",
    "Hungaroring": "hungary", "Spa-Francorchamps": "belgium", "Zandvoort": "netherlands",
    "Monza": "monza", "Baku": "baku", "Singapore": "singapore", "Austin": "usa",
    "Mexico City": "mexico", "Interlagos": "brazil", "Las Vegas": "lasvegas", "Lusail": "qatar",
    "Yas Marina Circuit": "abudhabi",
}  # fmt: skip
ACTION_TEXT = {
    "STAY": "Stay out",
    "PIT_SOFT": "Box · SOFT",
    "PIT_MEDIUM": "Box · MEDIUM",
    "PIT_HARD": "Box · HARD",
}
STATUS_CODE = {"SC": "4", "VSC": "6"}


def neutral_intervals(
    messages: list[dict[str, Any]], lap_s: float
) -> list[tuple[float, float, str]]:
    """SC/VSC periods from race control: DEPLOYED until ENDING / IN THIS LAP is resolved."""
    out: list[tuple[float, float, str]] = []
    start: tuple[float, str] | None = None
    for m in sorted(messages, key=lambda m: m.get("t") or 0.0):
        text, t = str(m.get("message") or "").upper(), float(m.get("t") or 0.0)
        if m.get("category") != "SafetyCar":
            continue
        kind = "VSC" if "VIRTUAL" in text or text.startswith("VSC") else "SC"
        if "DEPLOYED" in text and start is None:
            start = (t, kind)
        elif start is not None and ("ENDING" in text or "IN THIS LAP" in text):
            # VSC ends within seconds; an SC returns to the pits at the end of the lap.
            out.append((start[0], t + (15.0 if "ENDING" in text else lap_s), start[1]))
            start = None
    if start is not None:
        out.append((start[0], float("inf"), start[1]))
    return out


def field_laps(race: dict[str, Any]) -> tuple[list[FieldLap], dict[int, bool]]:
    """Public-timing field laps plus the field-level neutralised flag per lap."""
    rows = [r for laps in race.get("laps", {}).values() for r in laps if r[3]]
    lap_s = float(np.median([r[3] for r in rows])) if rows else 100.0
    periods = neutral_intervals(race.get("messages", []), lap_s)
    pits = {
        (int(p["driver_number"]), int(p["lap_number"]))
        for p in race.get("pits", [])
        if p.get("lap_number")
    }
    stints: dict[int, list[dict[str, Any]]] = {}
    for s in race.get("stints", []):
        stints.setdefault(int(s["driver_number"]), []).append(s)
    out: list[FieldLap] = []
    flagged: dict[int, list[bool]] = {}
    for key, laps in race.get("laps", {}).items():
        drv = int(key)
        for start, end, lap, duration, *_ in laps:
            stint = _stint(stints.get(drv, []), int(lap))
            status = "1"
            for a, b, kind in periods:
                if end is not None and start < b and end > a:
                    status = STATUS_CODE[kind]
            flagged.setdefault(int(lap), []).append(status != "1")
            compound = str(stint.get("compound") or "").upper() if stint else ""
            if compound not in (*COMPOUNDS, "INTERMEDIATE", "WET"):
                continue  # no stint record: the lap's tyre is unknown, so it is not used
            age = (
                int(stint.get("tyre_age_at_start") or 0) + int(lap) - int(stint["lap_start"]) + 1
                if stint
                else None
            )
            pit_in, pit_out = (drv, int(lap)) in pits, (drv, int(lap) - 1) in pits
            out.append(
                FieldLap(
                    driver_number=str(drv),
                    lap_number=int(lap),
                    lap_time_s=duration,
                    compound=compound,
                    tyre_life_laps=age,
                    track_status=status,
                    pit_in=pit_in,
                    pit_out=pit_out,
                    is_accurate=bool(duration)
                    and compound in COMPOUNDS
                    and not (pit_in or pit_out),
                )
            )
    neutral = {lap: float(np.mean(v)) >= 0.3 for lap, v in flagged.items()}
    return out, neutral


def _stint(stints: list[dict[str, Any]], lap: int) -> dict[str, Any] | None:
    hits = [
        s
        for s in stints
        if s.get("lap_start") and s["lap_start"] <= lap and (s.get("lap_end") or 10**6) >= lap
    ]
    return max(hits, key=lambda s: s["lap_start"]) if hits else None


def advisor_records(
    race: dict[str, Any], advisor: StrategyAdvisor, engine: Literal["advisor", "ensemble"]
) -> dict[str, Any]:
    """Component payload with one call per driver per lap (replay)."""
    mode: Literal["planner", "ensemble"] = "ensemble" if engine == "ensemble" else "planner"
    meta = race.get("meta", {})
    circuit = CIRCUITS.get(str(meta.get("circuit_short_name")), str(meta.get("circuit_short_name")))
    total = int(race["total_laps"])
    laps, neutral = field_laps(race)
    by_driver: dict[str, list[FieldLap]] = {}
    for f in laps:
        by_driver.setdefault(f.driver_number, []).append(f)
    ends = {(int(k), int(r[2])): r[1] for k, rows in race.get("laps", {}).items() for r in rows}
    est_cache: dict[int, tuple[TyreParams, np.ndarray, Plan]] = {}
    label = "ensemble (opt-in)" if mode == "ensemble" else "planner"
    records = []
    for drv, rows in sorted(by_driver.items(), key=lambda kv: int(kv[0])):
        used: list[str] = []
        for row in sorted(rows, key=lambda r: r.lap_number):
            lap = row.lap_number
            base = {
                "driver": int(drv),
                "lap": lap,
                "completed_lap": lap - 1,
                "input_cutoff_lap": lap - 1,
                "input_cutoff_t": ends.get((int(drv), lap - 1)) if lap > 1 else 0.0,
                "model": f"Strategy advisor · {label}",
                "candidate_engine": "exact_dp_planner",
                "cost_units": "seconds",
            }
            if row.compound not in COMPOUNDS:
                records.append(
                    base
                    | {
                        "status": "unavailable",
                        "action": "unavailable",
                        "reason": "Wet tyres: the advisor covers dry races only.",
                    }
                )
                continue
            if row.compound not in used:
                used.append(row.compound)
            if lap not in est_cache:
                probe = CarState(
                    circuit=circuit,
                    total_laps=total,
                    lap=lap,
                    compound="MEDIUM",
                    tyre_age=1,
                    compounds_used=["MEDIUM"],
                )
                est, evidence = advisor.estimate(probe, [f for f in laps if f.lap_number < lap])
                est_cache[lap] = (est, evidence, solve(est, total))
            est, evidence, plan = est_cache[lap]
            state = CarState(
                circuit=circuit, total_laps=total, lap=lap, compound=row.compound,
                tyre_age=max(1, row.tyre_life_laps or 1), compounds_used=used,
                neutral=neutral.get(lap, False),
            )  # fmt: skip
            rec = advisor.decide(state, est, evidence, plan, mode)
            pit = rec.action != "STAY"
            records.append(
                base
                | {
                    "status": "ready",
                    "decision_valid": True,
                    "action": "pit_now" if pit else ("run_to_flag" if lap == total else "stay_out"),
                    "fit": rec.action.removeprefix("PIT_") if pit else None,
                    "stop_lap": lap if pit else None,
                    "model_version": rec.model["ensemble_candidate"]
                    if mode == "ensemble"
                    else f"priors {rec.model['priors']}",
                    "reason": rec.explanation,
                    "pit_loss_s": est.pit_neutral if state.neutral else est.pit_green,
                    "pit_saves_s": rec.pit_saves_s,
                    "ensemble_pit_probability": rec.ensemble_pit_probability,
                    "candidates": [
                        {
                            "label": ACTION_TEXT[k],
                            "cost_s": v,
                            "reason": "planner expected remaining time",
                        }
                        for k, v in rec.option_costs_s.items()
                    ],
                }
            )
    return {
        "backend": "strategy_advisor",
        "model_version": records[0].get("model_version", "") if records else "",
        "evaluation_mode": "retrospective_replay",
        "promoted": mode == "planner",
        "records": records,
        "coverage": {"drivers": len(by_driver), "rows": len(records), "circuit": circuit},
    }
