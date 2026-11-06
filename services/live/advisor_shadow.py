"""Live shadow mode for the strategy advisor: record calls silently, review after the race.

Live snapshots keep only each car's last two lap starts and no lap durations, so a
live advisor must accumulate history across polls. ``LiveFieldHistory`` rebuilds
completed lap times (``latest_completed`` and successive lap starts), stints and
SC/VSC status from the stream; ``AdvisorShadow`` issues each car's call for its
next lap once, using only what had arrived by then, and journals it. Nothing is
shown to users. ``review`` compares a journal with the post-race replay calls;
that comparison is the evidence required before live mode can be enabled.

    python -m services.live.advisor_shadow record --snapshots data/live/11396/snapshots.jsonl \
        --out data/live/11396/advisor-shadow.jsonl
    python -m services.live.advisor_shadow watch --out data/live/<session>/advisor-shadow.jsonl
    python -m services.live.advisor_shadow review --shadow data/live/11396/advisor-shadow.jsonl \
        --replay data/race_center/race-11396-v3.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ml.strategy.planner import solve
from ml.strategy_serving.advisor import COMPOUNDS, CarState, FieldLap, StrategyAdvisor

from services.live.advisor_strategy import CIRCUITS, advisor_records

NEUTRAL_CODES = {"4": "4", "6": "6", "7": "6"}  # SC, VSC deployed, VSC ending
SHADOW_VERSION = "advisor-live-shadow-v1"
# Measured on the 2026 US GP capture: new tyres appear in the stint feed 26-37 s
# after the out-lap starts. Calls wait this long into the lap (~half a lap).
STINT_FEED_HOLD_S = 45.0


@dataclass
class _Driver:
    starts: dict[int, float] = field(default_factory=dict)
    durations: dict[int, float] = field(default_factory=dict)
    # stint_number -> {compound, start, anchor, anchor_age, rank}
    # age(lap) = anchor_age + lap - anchor + 1
    stints: dict[int, dict[str, Any]] = field(default_factory=dict)


class LiveFieldHistory:
    """Accumulates what live timing has published so far; never reads ahead."""

    def __init__(self) -> None:
        self.drivers: dict[int, _Driver] = {}
        self.status: list[tuple[float, str]] = []
        self.meta: dict[str, Any] = {}
        self.total_laps: int | None = None
        self.t = 0.0

    def ingest(self, snap: dict[str, Any]) -> None:
        self.meta = snap.get("meta") or self.meta
        self.t = float(snap.get("t") or self.t)
        if snap.get("distance_confirmed") and snap.get("total_laps"):
            self.total_laps = int(snap["total_laps"])
        self.status.append((self.t, str(snap.get("track_status") or "1")))
        for key, rows in (snap.get("laps") or {}).items():
            d = self.drivers.setdefault(int(key), _Driver())
            for row in rows:
                d.starts.setdefault(int(row[2]), float(row[0]))
                if row[3]:
                    d.durations[int(row[2])] = float(row[3])
        for key, row in (snap.get("latest_completed") or {}).items():
            if row and row[3]:
                self.drivers.setdefault(int(key), _Driver()).durations[int(row[2])] = float(row[3])
        for st in snap.get("stints") or []:
            d = self.drivers.setdefault(int(st["driver_number"]), _Driver())
            self._stint(d, st)
        for d in self.drivers.values():  # durations from consecutive published starts
            for lap, start in d.starts.items():
                if lap not in d.durations and lap + 1 in d.starts:
                    d.durations[lap] = round(d.starts[lap + 1] - start, 3)

    @staticmethod
    def _stint(d: _Driver, st: dict[str, Any]) -> None:
        """Normalise a published stint. Live records often carry lap_start 0 (unknown)
        with the set's current age, so anchor on (lap in progress, age) instead."""
        compound = str(st.get("compound") or "").upper()
        number = int(st.get("stint_number") or 0)
        if not compound or number < 1:
            return
        start = int(st.get("lap_start") or 0)
        # Trust: closed records (exact) > current samples (lap in progress, age at its
        # start) > lap_start 0 with an age (the lap-row clock can lag the stint feed).
        if start >= 1 and not st.get("current_sample"):
            rank = 3
            rec = {"compound": compound, "start": start, "anchor": start,
                   "anchor_age": int(st.get("tyre_age_at_start") or 0)}  # fmt: skip
        elif start >= 1:
            rank, age = 2, int(float(st.get("tyre_age_at_start") or 0))
            rec = {"compound": compound, "start": start - age, "anchor": start, "anchor_age": age}
        elif st.get("tyre_age") is not None and d.starts:
            rank, cur, age = 1, max(d.starts), int(float(st["tyre_age"]))
            rec = {"compound": compound, "start": cur - age, "anchor": cur, "anchor_age": age}
        else:
            return
        rec["rank"] = rank
        old = d.stints.get(number)
        if old is None or rank >= old["rank"]:  # newest observation wins within a rank
            d.stints[number] = rec

    @staticmethod
    def _stint_at(d: _Driver, lap: int) -> dict[str, Any] | None:
        hit = [s for s in d.stints.values() if s["start"] <= lap]
        return max(hit, key=lambda s: s["start"]) if hit else None

    @staticmethod
    def _age(stint: dict[str, Any], lap: int) -> int:
        return max(1, int(stint["anchor_age"]) + lap - int(stint["anchor"]) + 1)

    def _neutral(self, start: float, end: float) -> str:
        codes = [
            NEUTRAL_CODES[c] for t, c in self.status if start <= t <= end and c in NEUTRAL_CODES
        ]
        return codes[0] if codes else "1"

    def field_laps(self) -> list[FieldLap]:
        out = []
        for num, d in self.drivers.items():
            stint_starts = {int(st["start"]) for st in d.stints.values() if st["start"] > 1}
            for lap, dur in sorted(d.durations.items()):
                stint = self._stint_at(d, lap)
                comp = str(stint.get("compound") or "").upper() if stint else ""
                if comp not in (*COMPOUNDS, "INTERMEDIATE", "WET"):
                    continue
                start = d.starts.get(lap)
                status = self._neutral(start, start + dur) if start is not None else "1"
                age = self._age(stint, lap)  # type: ignore[arg-type]
                pit_out = lap in stint_starts and lap > 1
                pit_in = (lap + 1) in stint_starts
                out.append(
                    FieldLap(
                        driver_number=str(num),
                        lap_number=lap,
                        lap_time_s=dur,
                        compound=comp,
                        tyre_life_laps=max(1, age),
                        track_status=status,
                        pit_in=pit_in,
                        pit_out=pit_out,
                        is_accurate=comp in COMPOUNDS and not (pit_in or pit_out) and lap > 1,
                    )
                )
        return out


class AdvisorShadow:
    """Issues one silent call per car per lap, as soon as the previous lap is published."""

    def __init__(
        self, advisor: StrategyAdvisor, mode: Literal["planner", "ensemble"] = "planner"
    ) -> None:
        self.advisor, self.mode = advisor, mode
        self.history = LiveFieldHistory()
        self.issued: set[tuple[int, int]] = set()

    def step(self, snap: dict[str, Any], recorded_at: str | None = None) -> list[dict[str, Any]]:
        h = self.history
        h.ingest(snap)
        if h.total_laps is None:
            return []  # never advise without a confirmed race distance
        circuit = CIRCUITS.get(
            str(h.meta.get("circuit_short_name")), str(h.meta.get("circuit_short_name"))
        )
        laps = h.field_laps()
        cache: dict[int, Any] = {}
        calls = []
        for num, d in h.drivers.items():
            if not d.durations:
                continue
            lap = max(d.durations) + 1
            if lap > h.total_laps or (num, lap) in self.issued:
                continue
            started = d.starts.get(lap)
            if started is None or h.t - started < STINT_FEED_HOLD_S:
                continue  # after a stop the stint feed reports new tyres 26-37 s into the lap
            stint = h._stint_at(d, lap)
            comp = str((stint or {}).get("compound") or "").upper()
            if comp not in COMPOUNDS:
                continue
            used = sorted(
                {
                    st["compound"]
                    for st in d.stints.values()
                    if st["start"] <= lap and st["compound"] in COMPOUNDS
                }
                | {comp}
            )
            if lap not in cache:
                probe = CarState(
                    circuit=circuit,
                    total_laps=h.total_laps,
                    lap=lap,
                    compound="MEDIUM",
                    tyre_age=1,
                    compounds_used=["MEDIUM"],
                )
                est, evidence = self.advisor.estimate(
                    probe, [f for f in laps if f.lap_number < lap]
                )
                cache[lap] = (est, evidence, solve(est, h.total_laps))
            est, evidence, plan = cache[lap]
            age = self.history._age(stint, lap)  # type: ignore[arg-type]
            neutral = h.status[-1][1] in NEUTRAL_CODES
            state = CarState(
                circuit=circuit, total_laps=h.total_laps, lap=lap, compound=comp,
                tyre_age=max(1, age), compounds_used=used, neutral=neutral,
            )  # fmt: skip
            rec = self.advisor.decide(state, est, evidence, plan, self.mode)
            self.issued.add((num, lap))
            calls.append(
                {
                    "version": SHADOW_VERSION,
                    "session_key": h.meta.get("session_key"),
                    "driver": num,
                    "lap": lap,
                    "t": h.t,
                    "recorded_at": recorded_at or datetime.now(UTC).isoformat(),
                    "action": rec.action,
                    "planner_action": rec.planner_action,
                    "pit_saves_s": rec.pit_saves_s,
                    "compound": comp,
                    "tyre_age": state.tyre_age,
                    "neutral": neutral,
                    "field_laps_used": sum(1 for f in laps if f.lap_number < lap),
                    "mode": self.mode,
                }
            )
        return calls


def _snapshots(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def record(shadow: AdvisorShadow, snaps: Iterator[dict[str, Any]], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("a", encoding="utf-8") as f:
        for snap in snaps:
            for call in shadow.step(snap, recorded_at=snap.get("updated_at")):
                f.write(json.dumps(call) + "\n")
                n += 1
    return n


def review(
    shadow_path: Path,
    replay_path: Path,
    advisor: StrategyAdvisor,
    gate_config: Path | None = Path("configs/strategy-live-shadow-v1.json"),
) -> dict[str, Any]:
    """Compare live shadow calls with post-race replay calls for the same car and lap."""
    shadow = [json.loads(x) for x in shadow_path.read_text().splitlines() if x.strip()]
    race = json.loads(replay_path.read_text())
    replay = {
        (r["driver"], r["lap"]): r
        for r in advisor_records(race, advisor, "advisor")["records"]
        if r["status"] == "ready"
    }
    pairs = [
        (s, replay[(s["driver"], s["lap"])]) for s in shadow if (s["driver"], s["lap"]) in replay
    ]
    if not pairs:
        return {"shadow_calls": len(shadow), "matched": 0}
    live_pit = np.array([s["action"] != "STAY" for s, _ in pairs])
    post_pit = np.array([r["action"] == "pit_now" for _, r in pairs])
    same = live_pit == post_pit
    same_full = [
        s["action"] == ("STAY" if r["action"] != "pit_now" else "PIT_" + str(r["fit"]))
        for s, r in pairs
    ]
    delta = np.array([abs(s["pit_saves_s"] - r["pit_saves_s"]) for s, r in pairs])
    expected = {k for k in replay if k[1] > 1}
    got = {(s["driver"], s["lap"]) for s in shadow}
    first: dict[int, int] = {}
    for d, lap in got:
        first[d] = min(first.get(d, lap), lap)
    reachable = {k for k in expected if k[1] >= first.get(k[0], 10**6)}
    replay_saves = np.array([r["pit_saves_s"] for _, r in pairs])
    clear = np.abs(replay_saves) >= 1.0
    fresh_pits = int(np.sum(live_pit & (replay_saves < -10.0)))
    clear_agree = float(same[clear].mean()) if clear.any() else 0.0
    reach = len(got & reachable) / max(1, len(reachable))
    gates = {}
    if gate_config is not None and gate_config.exists():
        g = json.loads(gate_config.read_text())["qualification"]["per_race_gates"]
        gates = {
            "clear_call_agreement": clear_agree >= g["clear_call_agreement_min"],
            "no_pit_on_just_fitted_tyres": fresh_pits <= g["pit_on_just_fitted_tyres_max"],
            "reachable_coverage": reach >= g["reachable_coverage_min"],
        }
    return {
        "shadow_calls": len(shadow),
        "matched_to_replay": len(pairs),
        "coverage_of_replay_car_laps": round(len(got & expected) / max(1, len(expected)), 4),
        "reachable_coverage": round(reach, 4),
        "same_pit_or_stay": round(float(same.mean()), 4),
        "clear_call_agreement": round(clear_agree, 4),
        "clear_calls": int(clear.sum()),
        "pit_on_just_fitted_tyres": fresh_pits,
        "same_exact_action": round(float(np.mean(same_full)), 4),
        "pit_calls_live": int(live_pit.sum()),
        "pit_calls_replay": int(post_pit.sum()),
        "pit_saves_abs_diff_s": {
            "median": round(float(np.median(delta)), 3),
            "p95": round(float(np.percentile(delta, 95)), 3),
        },
        "gates": gates,
        "passes": bool(gates) and all(gates.values()),
        "disagreements": [
            {
                "driver": s["driver"],
                "lap": s["lap"],
                "live": s["action"],
                "replay": r["action"],
                "live_saves": s["pit_saves_s"],
                "replay_saves": r["pit_saves_s"],
            }
            for (s, r), ok in zip(pairs, same, strict=True) if not ok
        ][:50],
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=["record", "watch", "review"])
    ap.add_argument("--snapshots", type=Path)
    ap.add_argument("--snapshot-file", type=Path, default=Path("data/live/free-timing.json"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--shadow", type=Path)
    ap.add_argument("--replay", type=Path)
    ap.add_argument("--mode", choices=["planner", "ensemble"], default="planner")
    ap.add_argument("--interval", type=float, default=10.0)
    args = ap.parse_args()
    advisor = StrategyAdvisor.from_config()
    if args.command == "review":
        print(json.dumps(review(args.shadow, args.replay, advisor), indent=2))
        return
    shadow = AdvisorShadow(advisor, args.mode)
    if args.command == "record":
        print(f"{record(shadow, _snapshots(args.snapshots), args.out)} shadow calls -> {args.out}")
        return
    last = None
    while True:  # watch: poll the collector's atomic snapshot; Ctrl+C to stop
        try:
            snap = json.loads(args.snapshot_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            snap = None
        if snap and snap.get("updated_at") != last:
            last = snap.get("updated_at")
            n = record(shadow, iter([snap]), args.out)
            if n:
                print(f"{datetime.now(UTC):%H:%M:%S} recorded {n} call(s)", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
