"""Measure the deployed in-race estimator's error on training races (P1).

For each training race and each decision lap, the estimate the planner/policy
would have had (prior + field laps through the previous lap) is compared with the
race's hindsight parameters. The resulting trajectories are replayed by
``StrategyEnv(errors=...)`` so training sees realistic, lap-dependent estimation
error instead of a fixed lognormal guess. Only training races may be used.

    python -m ml.strategy.estimation_error --out data/estimation_error_train.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ml.strategy.env import ErrorTrajectory
from ml.strategy.race_model import DRY, clean_laps, inrace_update, prerace_prior


def measure(universe: object, keys: list[tuple[int, int]]) -> list[ErrorTrajectory]:
    """One trajectory per race in ``keys`` (pass ``universe.splits['train']``)."""
    races, truths = universe.races, universe.truths  # type: ignore[attr-defined]
    out = []
    for k in keys:
        race, truth = races[k], truths[k]
        prior = prerace_prior(truths, races, race)
        n, tv = race.n_laps, truth.as_vector()
        prog, tyre, evid = [], [], []
        for lap in range(1, n + 1):
            est = inrace_update(race, prior, lap - 1) if lap > 5 else prior
            seen = clean_laps(race, lap - 1) if lap > 1 else race.laps.iloc[:0]
            evid.append([int((seen["compound_encoded"] == c).sum()) for c in DRY])
            tyre.append(est.as_vector() - tv)
            prog.append(lap / n)
        out.append(
            ErrorTrajectory(
                progress=np.asarray(prog),
                tyre=np.asarray(tyre),
                evidence=np.asarray(evid, dtype=float),
                pit_green=prior.pit_green - truth.pit_green,
                pit_neutral=prior.pit_neutral - truth.pit_neutral,
                p_neutral=prior.p_neutral - truth.p_neutral,
            )
        )
    return out


def save(trajs: list[ErrorTrajectory], path: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    for i, t in enumerate(trajs):
        arrays[f"{i}_progress"], arrays[f"{i}_tyre"], arrays[f"{i}_evidence"] = (
            t.progress,
            t.tyre,
            t.evidence,
        )
        arrays[f"{i}_scalars"] = np.array([t.pit_green, t.pit_neutral, t.p_neutral])
    np.savez_compressed(path, count=np.array(len(trajs)), **arrays)


def load(path: Path) -> list[ErrorTrajectory]:
    z = np.load(path)
    out = []
    for i in range(int(z["count"])):
        s = z[f"{i}_scalars"]
        out.append(
            ErrorTrajectory(
                z[f"{i}_progress"],
                z[f"{i}_tyre"],
                z[f"{i}_evidence"],
                float(s[0]),
                float(s[1]),
                float(s[2]),
            )
        )
    return out


def main() -> None:
    from ml.strategy.protocol import CONFIG, load_universe

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/strategy_gold_2021_2026.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/estimation_error_train.npz"))
    args = ap.parse_args()
    u = load_universe(args.data, json.loads(CONFIG.read_text()))
    trajs = measure(u, u.splits["train"])
    save(trajs, args.out)
    deg = np.concatenate([t.tyre[:, 2:5] for t in trajs])
    rms = float(np.sqrt((deg**2).mean()))
    print(f"{len(trajs)} training races -> {args.out}; deg error RMS {rms:.4f} s/lap")


if __name__ == "__main__":
    main()
