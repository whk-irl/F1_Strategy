"""Chronological held-out evaluation of pit policies against real team strategies.

For each dry, red-flag-free race the *truth* model is fitted in hindsight from
that race's complete laps. Every strategy for a driver (the team's real one, the
planner's, heuristics, oracle) is scored in that same model with the race's real
SC/VSC timeline, so differences come only from the strategy.

The planner itself sees only: priors from strictly earlier races, field laps
completed before the decision, and whether the current lap is neutralised.

Limitations (stated in outputs): no traffic/track-position model, new sets
assumed fresh, starting compound fixed to the real one.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.strategy.planner import Plan, solve, solve_known_timeline
from ml.strategy.race_model import (
    DRY,
    Race,
    TyreParams,
    empirical_bayes_truths,
    inrace_update,
    prepare_race,
    prerace_prior,
)

Policy = Callable[[int, int, int, bool, bool], int]  # lap, compound, age, two_used, neutral


def actual_strategy(race: Race, driver: str) -> tuple[int, int, dict[int, int]] | None:
    """(start compound, start age, {pit lap: new compound}) or None if unusable."""
    d = race.laps[race.laps["driver_number"].astype(str) == driver].sort_values("lap_number")
    if d.empty or not d["compound_encoded"].isin(DRY).all():
        return None
    by_lap = d.set_index("lap_number")
    first = by_lap.iloc[0]
    stops: dict[int, int] = {}
    for lap in by_lap.index[by_lap["pit_in"]]:
        if lap + 1 in by_lap.index:
            stops[int(lap)] = int(by_lap.at[lap + 1, "compound_encoded"])
    start_age = int(first["tyre_life_laps"]) if np.isfinite(first["tyre_life_laps"]) else 1
    return int(first["compound_encoded"]), max(1, start_age), stops


def score(
    truth: TyreParams,
    race: Race,
    n_laps: int,
    start: tuple[int, int],
    policy: Policy,
) -> tuple[float, int, list[tuple[int, int]]]:
    """Strategy-dependent race time in the truth model; returns (time, stops, plan)."""
    c, a = start
    used = {c}
    total = 0.0
    stops: list[tuple[int, int]] = []
    for lap in range(1, n_laps + 1):
        neutral = bool(race.neutral_lap[lap])
        if lap > 1 and not neutral:  # lap 1 is identical for every strategy
            total += float(truth.stint_cost(c, a))
        if lap == n_laps:
            break
        act = policy(lap, c, a, len(used) >= 2, neutral)
        if act:
            c2 = act - 1
            total += truth.pit_neutral if neutral else truth.pit_green
            stops.append((lap, c2))
            c, a = c2, 1
            used.add(c2)
        else:
            a += 1
    if len(used) < 2:
        total += 1e4
    return total, len(stops), stops


def from_schedule(stops: dict[int, int]) -> Policy:
    return lambda lap, c, a, f, s: (stops[lap] + 1) if lap in stops else 0


def planner_policy(
    race: Race, prior: TyreParams, n_laps: int, cache: dict[int, Plan], update: bool
) -> Policy:
    def act(lap: int, c: int, a: int, f: bool, s: bool) -> int:
        key = lap if update else 0
        if key not in cache:
            params = inrace_update(race, prior, lap - 1) if update and lap > 5 else prior
            cache[key] = solve(params, n_laps)
        return cache[key].act(lap, c, a, f, s)

    return act


def heuristic_one_stop(n_laps: int) -> Policy:
    """Common baseline: one stop at mid-race to the hardest unused compound."""
    mid = n_laps // 2

    def act(lap: int, c: int, a: int, f: bool, s: bool) -> int:
        if lap == mid and not f:
            return 3 if c != 2 else 2
        return 0

    return act


def heuristic_sc_opportunist(n_laps: int) -> Policy:
    """One-stop baseline that takes an SC/VSC stop inside the 30-70% window."""
    mid = n_laps // 2
    done = {"x": False}

    def act(lap: int, c: int, a: int, f: bool, s: bool) -> int:
        if f:
            return 0
        if (s and 0.3 * n_laps <= lap <= 0.7 * n_laps) or lap == mid:
            return 3 if c != 2 else 2
        return 0

    _ = done
    return act


def evaluate(df: pd.DataFrame, eval_years: tuple[int, ...]) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = sorted(
        {(int(y), int(r)) for y, r in df[["year", "round_number"]].itertuples(index=False)}
    )
    races = {k: prepare_race(df, *k) for k in keys}
    usable = {k: r for k, r in races.items() if not r.wet and not r.red_flag}
    truths = empirical_bayes_truths(usable)

    records = []
    for k, race in usable.items():
        if k[0] not in eval_years:
            continue
        truth = truths[k]
        prior = prerace_prior(dict(truths.items()), usable, race)
        prior_cache: dict[int, Plan] = {}
        live_cache: dict[int, Plan] = {}
        oracle = solve_known_timeline(truth, race.neutral_lap, race.n_laps)
        for driver in sorted(race.laps["driver_number"].astype(str).unique()):
            strat = actual_strategy(race, driver)
            nd = int(
                race.laps.loc[race.laps["driver_number"].astype(str) == driver, "lap_number"].max()
            )
            if strat is None or nd < race.n_laps - 1:
                continue
            c0, a0, stops = strat
            if len({c0, *stops.values()}) < 2:
                continue
            n = nd
            if n != race.n_laps:
                prior_c: dict[int, Plan] = {}
                live_c: dict[int, Plan] = {}
                orc = solve_known_timeline(truth, race.neutral_lap, n)
            else:
                prior_c, live_c, orc = prior_cache, live_cache, oracle
            team, team_n, _ = score(truth, race, n, (c0, a0), from_schedule(stops))
            plan, plan_n, plan_stops = score(
                truth, race, n, (c0, a0), planner_policy(race, prior, n, live_c, True)
            )
            pre, _pre_n, _ = score(
                truth, race, n, (c0, a0), planner_policy(race, prior, n, prior_c, False)
            )
            h1, _, _ = score(truth, race, n, (c0, a0), heuristic_one_stop(n))
            h2, _, _ = score(truth, race, n, (c0, a0), heuristic_sc_opportunist(n))
            orac, orac_n, _ = score(truth, race, n, (c0, a0), orc.act)
            records.append(
                {
                    "year": k[0],
                    "round": k[1],
                    "circuit": race.circuit,
                    "driver": driver,
                    "laps": n,
                    "team_s": team,
                    "planner_s": plan,
                    "prior_only_s": pre,
                    "one_stop_s": h1,
                    "sc_one_stop_s": h2,
                    "oracle_s": orac,
                    "team_stops": team_n,
                    "planner_stops": plan_n,
                    "oracle_stops": orac_n,
                    "planner_plan": json.dumps(plan_stops),
                    "team_plan": json.dumps(sorted(stops.items())),
                }
            )
    res = pd.DataFrame(records)
    summary = summarize(res)
    summary["races_evaluated"] = int(res[["year", "round"]].drop_duplicates().shape[0])
    summary["races_excluded_wet_or_red"] = int(
        sum(1 for k, r in races.items() if k[0] in eval_years and (r.wet or r.red_flag))
    )
    summary["truth_params_example"] = {
        f"{k[0]}-{k[1]}": asdict(v) for k, v in list(truths.items())[:3]
    }
    return res, summary


def summarize(res: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"driver_races": len(res)}
    for name in ("planner_s", "prior_only_s", "one_stop_s", "sc_one_stop_s", "oracle_s"):
        diff = res["team_s"] - res[name]  # positive = faster than the real team
        out[name.removesuffix("_s")] = {
            "mean_gain_vs_team_s": round(float(diff.mean()), 2),
            "median_gain_vs_team_s": round(float(diff.median()), 2),
            "share_at_least_as_fast_as_team": round(float((diff >= -0.05).mean()), 3),
            "mean_regret_vs_oracle_s": round(float((res[name] - res["oracle_s"]).mean()), 2),
        }
    out["team"] = {
        "mean_regret_vs_oracle_s": round(float((res["team_s"] - res["oracle_s"]).mean()), 2)
    }
    out["stops"] = {
        "team_mean": round(float(res["team_stops"].mean()), 2),
        "planner_mean": round(float(res["planner_stops"].mean()), 2),
        "oracle_mean": round(float(res["oracle_stops"].mean()), 2),
        "planner_zero_stop_share": round(float((res["planner_stops"] == 0).mean()), 3),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/gold_2022_2025.parquet")
    ap.add_argument("--years", default="2024,2025")
    ap.add_argument("--out", default="runs/strategy-planner-eval")
    args = ap.parse_args()
    df = pd.read_parquet(args.data)
    res, summary = evaluate(df, tuple(int(y) for y in args.years.split(",")))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "per_driver.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "truth_params_example"}, indent=2))


if __name__ == "__main__":
    main()
