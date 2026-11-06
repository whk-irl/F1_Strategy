"""Frozen evaluation protocol for strategy candidates (configs/strategy-training-loop-v1.json).

A candidate is a directory with ``seed-*/policy.zip`` (optionally ``candidate.json``
naming its observation version). It is scored as a seed ensemble and per seed,
paired against the DP planner, real team strategies and the hindsight oracle.

    python -m ml.strategy.protocol evaluate --candidate runs/strategy-ppo-186536 --split val
    python -m ml.strategy.protocol evaluate --candidate ... --split stress
    python -m ml.strategy.protocol freeze   --candidate ...
    python -m ml.strategy.protocol evaluate --candidate ... --split test  # once per candidate

Splits: ``val`` = validation years; ``stress`` = each validation race replayed under
degradation/pit-loss perturbations with resampled SC timelines (same races, more
stable comparison, not independent evidence); ``test`` = single use, requires a
frozen candidate, and every access is appended to the test ledger before scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.strategy import evaluate as ev
from ml.strategy.env import observation, observation_v2
from ml.strategy.planner import Plan, solve_known_timeline
from ml.strategy.race_model import (
    Race,
    TyreParams,
    clean_laps,
    empirical_bayes_truths,
    inrace_update,
    prepare_race,
    prerace_prior,
)

CONFIG = Path("configs/strategy-training-loop-v1.json")
LEDGER = Path("runs/strategy-test-ledger.jsonl")
OBSERVERS: dict[str, Callable[..., np.ndarray]] = {
    "v1": lambda lap, n, c, a, f, s, est, evidence: observation(lap, n, c, a, f, s, est),
    "v2": observation_v2,
}
Act = Callable[[np.ndarray], int]


# ---------------------------------------------------------------------------
# Data and splits
# ---------------------------------------------------------------------------


@dataclass
class Universe:
    races: dict[tuple[int, int], Race]
    truths: dict[tuple[int, int], TyreParams]
    splits: dict[str, list[tuple[int, int]]]


def load_universe(data: Path, config: dict[str, Any]) -> Universe:
    df = pd.read_parquet(data)
    keys = sorted(
        {(int(y), int(r)) for y, r in df[["year", "round_number"]].itertuples(index=False)}
    )
    races = {k: prepare_race(df, *k) for k in keys}
    races = {k: r for k, r in races.items() if not r.wet and not r.red_flag}
    sp = config["splits"]
    lo, hi = sp["test"]["rounds"]
    splits = {
        "train": [k for k in races if k[0] in sp["train"]["years"]],
        "val": [k for k in races if k[0] in sp["validation"]["years"]],
        "test": [k for k in races if k[0] == sp["test"]["year"] and lo <= k[1] <= hi],
    }
    assert_disjoint(splits)
    truths = empirical_bayes_truths(races, pool_keys=set(splits["train"]))
    return Universe(races=races, truths=truths, splits=splits)


def assert_disjoint(splits: dict[str, list[tuple[int, int]]]) -> None:
    seen: dict[tuple[int, int], str] = {}
    for name, keys in splits.items():
        for k in keys:
            if k in seen:
                raise ValueError(f"Race {k} is in both {seen[k]} and {name}")
            seen[k] = name
    if any(k >= min(splits["val"], default=(9999, 0)) for k in splits["train"]):
        raise ValueError("Training races must precede validation races")


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    root: Path
    seeds: dict[str, Act]
    ensemble: Act
    observe: Callable[..., np.ndarray]


def load_candidate(root: Path) -> Candidate:
    from ml.strategy.ensemble import EnsemblePolicy

    meta = (
        json.loads((root / "candidate.json").read_text())
        if (root / "candidate.json").exists()
        else {}
    )
    paths = sorted(root.glob("seed-*/policy.zip"))
    if not paths:
        raise FileNotFoundError(f"No seed-*/policy.zip under {root}")
    ens = EnsemblePolicy.load(paths)
    seeds = {p.parent.name: _single(m) for p, m in zip(paths, ens.models, strict=True)}
    return Candidate(root, seeds, ens.act, OBSERVERS[meta.get("observation", "v1")])


def _single(model: Any) -> Act:
    return lambda o: int(model.predict(o, deterministic=True)[0])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class _Timeline:
    """Scoring view of a race with a replaced SC/VSC timeline."""

    def __init__(self, neutral: np.ndarray) -> None:
        self.neutral_lap = neutral


def resample_timeline(race: Race, rng: np.random.Generator, mean_len: float = 4.0) -> np.ndarray:
    p = float(race.neutral_lap[2:].mean()) if race.n_laps > 2 else 0.0
    stay, start = 1 - 1 / mean_len, p / (mean_len * max(1e-6, 1 - p))
    out, on = np.zeros_like(race.neutral_lap), False
    for lap in range(2, race.n_laps + 1):
        on = rng.random() < (stay if on else start)
        out[lap] = on
    return out


def score_split(u: Universe, cand: Candidate | None, split: str, replays: int = 5) -> pd.DataFrame:
    """Per car-race strategy times (seconds) for every policy on one split."""
    base_split = "val" if split == "stress" else split
    conditions: list[tuple[float, float, int | None]] = [(1.0, 1.0, None)]
    if split == "stress":
        conditions = [
            (d, p, i)
            for (d, p) in ((0.7, 1.0), (1.3, 1.0), (1.0, 0.8), (1.0, 1.2))
            for i in range(replays)
        ]
    rows = []
    for k in u.splits[base_split]:
        race, truth0 = u.races[k], u.truths[k]
        prior = prerace_prior(u.truths, u.races, race)
        n = race.n_laps
        est: dict[int, tuple[TyreParams, np.ndarray]] = {}
        plan_cache: dict[int, Plan] = {}
        drivers = []
        for d in sorted(race.laps["driver_number"].astype(str).unique()):
            st = ev.actual_strategy(race, d)
            nd = int(race.laps.loc[race.laps["driver_number"].astype(str) == d, "lap_number"].max())
            if st is not None and nd == n and len({st[0], *st[2].values()}) >= 2:
                drivers.append((d, st))
        for ci, (dscale, pscale, rep) in enumerate(conditions):
            truth = replace(
                truth0,
                deg=tuple(x * dscale for x in truth0.deg),  # type: ignore[arg-type]
                curv=tuple(x * dscale for x in truth0.curv),  # type: ignore[arg-type]
                pit_green=truth0.pit_green * pscale,
                pit_neutral=truth0.pit_neutral * pscale,
            )
            view: Any = race
            if rep is not None:
                view = _Timeline(
                    resample_timeline(race, np.random.default_rng(hash((k, rep)) % 2**32))
                )
            oracle = solve_known_timeline(truth, view.neutral_lap, n)
            for d, (c0, a0, stops) in drivers:
                start = (c0, a0)
                planner = ev.planner_policy(race, prior, n, plan_cache, True)
                row: dict[str, Any] = {
                    "year": k[0], "round": k[1], "circuit": race.circuit, "driver": d,
                    "condition": ci, "deg_scale": dscale, "pit_scale": pscale,
                    "team": ev.score(truth, view, n, start, ev.from_schedule(stops))[0],
                    "planner": ev.score(truth, view, n, start, planner)[0],
                    "oracle": ev.score(truth, view, n, start, oracle.act)[0],
                }  # fmt: skip
                if cand is not None:
                    pol = {"ensemble": cand.ensemble} | ({} if split == "stress" else cand.seeds)
                    for name, act in pol.items():
                        row[name] = ev.score(
                            truth, view, n, start, _driver(act, cand, race, prior, n, est)
                        )[0]
                rows.append(row)
    return pd.DataFrame(rows)


def _driver(
    act: Act,
    cand: Candidate,
    race: Race,
    prior: TyreParams,
    n: int,
    est: dict[int, tuple[TyreParams, np.ndarray]],
) -> ev.Policy:
    def policy(lap: int, c: int, a: int, f: bool, s: bool) -> int:
        if lap not in est:
            params = inrace_update(race, prior, lap - 1) if lap > 5 else prior
            seen = clean_laps(race, lap - 1) if lap > 1 else race.laps.iloc[:0]
            evidence = np.array([(seen["compound_encoded"] == k).sum() for k in (0, 1, 2)], float)
            est[lap] = (params, evidence)
        params, evidence = est[lap]
        return act(cand.observe(lap, n, c, a, f, s, params, evidence))

    return policy


# ---------------------------------------------------------------------------
# Statistics and gates
# ---------------------------------------------------------------------------


def paired(
    df: pd.DataFrame, col: str, ref: str = "planner", draws: int = 4000, seed: int = 0
) -> dict[str, Any]:
    """Mean (col - ref) per car-race with a race-cluster bootstrap 95% CI."""
    d = df.assign(race=df["year"] * 100 + df["round"], diff=df[col] - df[ref])
    g = d.groupby("race")["diff"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(draws, len(g)))
    s, c = g["sum"].to_numpy()[idx].sum(1), g["count"].to_numpy()[idx].sum(1)
    lo, hi = np.percentile(s / c, [2.5, 97.5])
    return {
        "mean_s": round(float(d["diff"].mean()), 3),
        "ci95_s": [round(float(lo), 3), round(float(hi), 3)],
        "races_better_share": round(float((g["sum"] < 0).mean()), 3),
        "races": len(g),
        "car_races": len(d),
    }


def summarize(df: pd.DataFrame, split: str, config: dict[str, Any]) -> dict[str, Any]:
    gates = config["release_gates"]
    out: dict[str, Any] = {
        "split": split,
        "planner_regret_s": round(float((df["planner"] - df["oracle"]).mean()), 3),
        "team_regret_s": round(float((df["team"] - df["oracle"]).mean()), 3),
    }
    if "ensemble" not in df:
        return out
    out["ensemble_vs_planner"] = paired(df, "ensemble")
    out["ensemble_regret_s"] = round(float((df["ensemble"] - df["oracle"]).mean()), 3)
    out["ensemble_illegal_share"] = float((df["ensemble"] >= 1e4).mean())
    seed_cols = [c for c in df.columns if c.startswith("seed-")]
    if seed_cols:
        means = {c: round(float((df[c] - df["planner"]).mean()), 3) for c in seed_cols}
        out["seed_minus_planner_s"] = means
        out["seed_spread_s"] = round(max(means.values()) - min(means.values()), 3)
        out["seed_illegal_share"] = {c: float((df[c] >= 1e4).mean()) for c in seed_cols}
    checks = {
        "legal": out["ensemble_illegal_share"] <= gates["illegal_share"],
        "noninferior": out["ensemble_vs_planner"]["ci95_s"][1] < gates["noninferiority_upper_ci_s"],
    }
    if split == "stress":
        worst = (
            df.assign(d=df["ensemble"] - df["planner"])
            .groupby(["deg_scale", "pit_scale"])["d"]
            .mean()
        )
        out["stress_by_condition_s"] = {
            f"deg{k[0]}_pit{k[1]}": round(float(v), 3) for k, v in worst.items()
        }
        checks["robust"] = bool(worst.max() <= gates["robustness_max_worse_than_planner_s"])
    out["gates"] = checks
    return out


# ---------------------------------------------------------------------------
# Freezing and the single-use test ledger
# ---------------------------------------------------------------------------


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(root: Path, data: Path) -> dict[str, Any]:
    code = sorted(Path(__file__).parent.glob("*.py"))
    files = sorted(root.glob("seed-*/policy.zip")) + [
        p for p in [root / "candidate.json"] if p.exists()
    ]
    return {
        "policies": {str(p.relative_to(root)): _sha(p) for p in files},
        "code": {p.name: _sha(p) for p in code},
        "data": _sha(data),
        "config": _sha(CONFIG),
    }


def freeze(root: Path, data: Path) -> dict[str, Any]:
    target = root / "FROZEN.json"
    if target.exists():
        raise FileExistsError(f"{target} exists; a frozen candidate is immutable")
    fp = fingerprint(root, data)
    fp["candidate_id"] = hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()[:16]
    target.write_text(json.dumps(fp, indent=2))
    return fp


def open_test(root: Path, data: Path, ledger: Path = LEDGER) -> str:
    """Verify the frozen candidate and record the single test access before scoring."""
    frozen_path = root / "FROZEN.json"
    if not frozen_path.exists():
        raise PermissionError("Test split requires `freeze` first")
    frozen = json.loads(frozen_path.read_text())
    now = fingerprint(root, data)
    for key in ("policies", "code", "data", "config"):
        if now[key] != frozen[key]:
            raise PermissionError(f"Candidate changed since freeze ({key}); freeze a new candidate")
    cid = str(frozen["candidate_id"])
    if ledger.exists():
        used = {
            json.loads(line)["candidate_id"]
            for line in ledger.read_text().splitlines()
            if line.strip()
        }
        if cid in used:
            raise PermissionError(f"Candidate {cid} already used the test split")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as f:
        stamp = pd.Timestamp.now(tz="UTC").isoformat()
        f.write(
            json.dumps({"candidate_id": cid, "candidate": str(root), "opened_utc": stamp}) + "\n"
        )
    return cid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=["evaluate", "freeze", "baseline"])
    ap.add_argument("--candidate", type=Path)
    ap.add_argument("--split", choices=["val", "stress", "test"], default="val")
    ap.add_argument("--data", type=Path, default=Path("data/strategy_gold_2021_2026.parquet"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--replays", type=int, default=5)
    args = ap.parse_args()
    config = json.loads(CONFIG.read_text())
    if args.command == "freeze":
        print(json.dumps(freeze(args.candidate, args.data), indent=2))
        return
    if args.split == "test":
        if args.command != "evaluate" or args.candidate is None:
            raise SystemExit("The test split is only for evaluating a frozen candidate")
        print("test access recorded for candidate", open_test(args.candidate, args.data))
    u = load_universe(args.data, config)
    cand = load_candidate(args.candidate) if args.command == "evaluate" else None
    df = score_split(u, cand, args.split, args.replays)
    report = summarize(df, args.split, config)
    out = (
        args.out
        or (args.candidate or Path("runs/strategy-planner-baseline")) / f"eval-{args.split}"
    )
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_car.csv", index=False)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
