"""Aggregate per-seed PPO benchmarks and apply the acceptance rule.

A seed qualifies when it never finishes illegally and is within ``--tolerance``
seconds of the DP planner on the held-out races. The serving verdict uses the
MEAN over seeds, never the best seed: picking the best seed on the same held-out
races that report its score is selection on test data. Serve the all-seed
ensemble (``ml.strategy.ensemble``) when every seed is legal and the seed mean
beats the planner; otherwise serve the planner.

    python -m ml.strategy.summarize_ppo runs/strategy-ppo-<jobid>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize(root: Path, tolerance: float) -> dict[str, Any]:
    seeds = []
    for path in sorted(root.glob("seed-*/benchmark.json")):
        r = json.loads(path.read_text())
        r["qualifies"] = r["ppo_illegal_share"] == 0 and r["ppo_minus_planner_s"] <= tolerance
        seeds.append(r)
    if not seeds:
        raise SystemExit(f"No seed-*/benchmark.json under {root}")
    best = min(seeds, key=lambda r: r["ppo_minus_planner_s"])
    gaps = [r["ppo_minus_planner_s"] for r in seeds]
    return {
        "seeds_found": len(seeds),
        "qualifying_seeds": [r["seed"] for r in seeds if r["qualifies"]],
        "ppo_minus_planner_s": {"min": min(gaps), "max": max(gaps), "mean": sum(gaps) / len(gaps)},
        "best_seed_on_test_do_not_select": best["seed"],
        "serve": "ppo_ensemble"
        if all(r["ppo_illegal_share"] == 0 for r in seeds) and sum(gaps) / len(gaps) < 0
        else "planner",
        "per_seed": seeds,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--tolerance", type=float, default=1.0)
    args = ap.parse_args()
    result = summarize(args.root, args.tolerance)
    (args.root / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_seed"}, indent=2))


if __name__ == "__main__":
    main()
