"""Build the serving prior artifact: tyre/pit priors per circuit from every completed race.

Uses the same chronological blending as evaluation (``race_model.prerace_prior``),
targeted after the last available race, so a future race at a known circuit gets
its circuit's history and a new venue gets the global prior.

    python -m ml.strategy_serving.priors --out models_baked/strategy_priors.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from ml.strategy.race_model import TyreParams, empirical_bayes_truths, prepare_race, prerace_prior

GLOBAL = "__global__"


def build(data: Path) -> dict[str, Any]:
    df = pd.read_parquet(data)
    keys = sorted(
        {(int(y), int(r)) for y, r in df[["year", "round_number"]].itertuples(index=False)}
    )
    races = {k: prepare_race(df, *k) for k in keys}
    races = {k: r for k, r in races.items() if not r.wet and not r.red_flag}
    truths = empirical_bayes_truths(races)
    after = (max(k[0] for k in races) + 1, 0)
    circuits = sorted({r.circuit for r in races.values()})

    def at(circuit: str) -> dict[str, Any]:
        target: Any = SimpleNamespace(key=after, circuit=circuit)
        return asdict(prerace_prior(truths, races, target))

    priors = {c: at(c) for c in circuits}
    priors[GLOBAL] = at(GLOBAL)
    return {
        "format_version": 1,
        "source_data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "races": len(races),
        "through": [int(x) for x in max(races)],
        "priors": priors,
    }


def load(path: Path) -> dict[str, TyreParams]:
    raw = json.loads(path.read_text())
    if raw.get("format_version") != 1:
        raise ValueError("Unsupported strategy prior artifact")
    return {
        c: TyreParams(
            off=(p["off"][0], p["off"][1], p["off"][2]),
            deg=(p["deg"][0], p["deg"][1], p["deg"][2]),
            curv=(p["curv"][0], p["curv"][1], p["curv"][2]),
            pit_green=p["pit_green"],
            pit_neutral=p["pit_neutral"],
            p_neutral=p["p_neutral"],
        )
        for c, p in raw["priors"].items()
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/strategy_gold_2021_2026.parquet"))
    ap.add_argument("--out", type=Path, default=Path("models_baked/strategy_priors.json"))
    args = ap.parse_args()
    art = build(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(art, indent=2))
    n = len(art["priors"]) - 1
    print(f"{n} circuits from {art['races']} races (through {art['through']}) -> {args.out}")


if __name__ == "__main__":
    main()
