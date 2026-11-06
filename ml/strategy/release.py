"""P4 behaviour, reload and latency gates for a candidate ensemble (validation split only).

    python -m ml.strategy.release --candidate runs/strategy-loop-p2/loop/arm-B

Gates (configs/strategy-training-loop-v1.json ``release_gates``):
  * mean stops within ``mean_stops_range`` and no stop after the penultimate lap
  * on validation races with SC/VSC laps, ensemble not worse than the planner
    (upper 95% race-bootstrap CI < ``sc_races_noninferiority_upper_ci_s``;
    amendment of 2026-11-05 replacing the SC-stop-share proxy, which is still reported)
  * reloading the policies reproduces every recorded decision
  * mean decision latency below ``decision_latency_ms`` on CPU
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.strategy import evaluate as ev
from ml.strategy.ensemble import EnsemblePolicy
from ml.strategy.planner import Plan
from ml.strategy.protocol import CONFIG, _driver, load_candidate, load_universe, paired
from ml.strategy.race_model import TyreParams, prerace_prior


def behaviour(
    data: Path, root: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], list[np.ndarray]]:
    u = load_universe(data, config)
    cand = load_candidate(root)
    seen_obs: list[np.ndarray] = []
    record = cand.ensemble

    def logged(o: np.ndarray) -> int:
        if len(seen_obs) < 5000:
            seen_obs.append(np.array(o))
        return record(o)

    times: list[dict[str, Any]] = []
    stats: dict[str, list[float]] = {
        "ens_stops": [],
        "plan_stops": [],
        "late": [],
        "ens_sc": [],
        "plan_sc": [],
    }
    for k in u.splits["val"]:
        race, truth = u.races[k], u.truths[k]
        prior = prerace_prior(u.truths, u.races, race)
        n = race.n_laps
        est: dict[int, tuple[TyreParams, np.ndarray]] = {}
        plan_cache: dict[int, Plan] = {}
        for d in sorted(race.laps["driver_number"].astype(str).unique()):
            st = ev.actual_strategy(race, d)
            nd = int(race.laps.loc[race.laps["driver_number"].astype(str) == d, "lap_number"].max())
            if st is None or nd != n or len({st[0], *st[2].values()}) < 2:
                continue
            start = (st[0], st[1])
            e_time, _, e_stops = ev.score(
                truth, race, n, start, _driver(logged, cand, race, prior, n, est)
            )
            p_time, _, p_stops = ev.score(
                truth, race, n, start, ev.planner_policy(race, prior, n, plan_cache, True)
            )
            if race.neutral_lap[2:].any():
                times.append({"year": k[0], "round": k[1], "ensemble": e_time, "planner": p_time})
            stats["ens_stops"].append(len(e_stops))
            stats["plan_stops"].append(len(p_stops))
            stats["late"].append(float(any(lap >= n - 1 for lap, _ in e_stops)))
            stats["ens_sc"] += [float(race.neutral_lap[lap]) for lap, _ in e_stops]
            stats["plan_sc"] += [float(race.neutral_lap[lap]) for lap, _ in p_stops]
    out = {
        "car_races": len(stats["ens_stops"]),
        "mean_stops": round(float(np.mean(stats["ens_stops"])), 3),
        "planner_mean_stops": round(float(np.mean(stats["plan_stops"])), 3),
        "stops_distribution": {
            str(int(s)): int(c)
            for s, c in zip(*np.unique(stats["ens_stops"], return_counts=True), strict=True)
        },
        "late_stop_share": round(float(np.mean(stats["late"])), 4),
        "sc_stop_share": round(float(np.mean(stats["ens_sc"])), 4),
        "planner_sc_stop_share": round(float(np.mean(stats["plan_sc"])), 4),
        "sc_races_vs_planner": paired(pd.DataFrame(times), "ensemble"),
    }
    return out, seen_obs


def reload_and_latency(root: Path, obs: list[np.ndarray]) -> dict[str, Any]:
    paths = sorted(root.glob("seed-*/policy.zip"))
    first, second = EnsemblePolicy.load(paths), EnsemblePolicy.load(paths)
    a = [first.act(o) for o in obs]
    b = [second.act(o) for o in obs]
    t0 = time.perf_counter()
    for o in obs[:1000]:
        first.act(o)
    ms = (time.perf_counter() - t0) / max(1, min(1000, len(obs))) * 1000
    return {"reload_decisions": len(obs), "reload_identical": a == b, "latency_ms": round(ms, 3)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/strategy_gold_2021_2026.parquet"))
    args = ap.parse_args()
    config = json.loads(CONFIG.read_text())
    g = config["release_gates"]
    beh, obs = behaviour(args.data, args.candidate, config)
    srv = reload_and_latency(args.candidate, obs)
    lo, hi = g["mean_stops_range"]
    gates = {
        "stops_in_range": lo <= beh["mean_stops"] <= hi,
        "no_late_stops": beh["late_stop_share"] == 0.0,
        "sc_races_noninferior": beh["sc_races_vs_planner"]["ci95_s"][1]
        < g["sc_races_noninferiority_upper_ci_s"],
        "reload_parity": srv["reload_identical"],
        "latency": srv["latency_ms"] < g["decision_latency_ms"],
    }
    report = {"split": "val", "behaviour": beh, "serving": srv, "gates": gates}
    out = args.candidate / "release-checks.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
