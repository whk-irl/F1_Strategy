"""Pre-registered prospective check of the frozen ensemble (configs/strategy-prospective-v1.json).

    python -m ml.strategy_serving.prospective status                  # interim, monitoring only
    python -m ml.strategy_serving.prospective decide --final-round 24 # once, after the season

Before scoring, every guard in the registration must hold: same frozen weights,
same ``ml/strategy`` code and training-loop config, same history rows (seasons
before 2026 and 2026 r1-18) and the same saved test rows. Each run is appended to
``runs/strategy-prospective-ledger.jsonl``; ``decide`` refuses to run twice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml.strategy import protocol

REGISTRATION = Path("configs/strategy-prospective-v1.json")
LEDGER = Path("runs/strategy-prospective-ledger.jsonl")


def history_sha(df: pd.DataFrame) -> str:
    sub = df[(df["year"] < 2026) | ((df["year"] == 2026) & (df["round_number"] <= 18))]
    sub = sub.sort_values(["year", "round_number", "driver_number", "lap_number"]).reset_index(
        drop=True
    )
    return hashlib.sha256(pd.util.hash_pandas_object(sub, index=False).values.tobytes()).hexdigest()


def verify(reg: dict[str, Any], data: Path, candidate: Path) -> list[str]:
    problems = []
    frozen = json.loads(Path(reg["frozen_record"]).read_text())
    now = protocol.fingerprint(candidate, data)
    if frozen["candidate_id"] != reg["candidate_id"]:
        problems.append("frozen record is a different candidate")
    for key in ("policies", "code", "config"):
        if now[key] != frozen[key]:
            problems.append(f"{key} changed since freeze")
    if history_sha(pd.read_parquet(data)) != reg["unchanged_history_sha256"]:
        problems.append("history rows (before 2026 r19) changed")
    test = Path(reg["test_rows"]["path"])
    if hashlib.sha256(test.read_bytes()).hexdigest() != reg["test_rows"]["sha256"]:
        problems.append("saved test rows changed")
    return problems


def _log(entry: dict[str, Any], ledger: Path) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def run(command: str, data: Path, final_round: int | None, ledger: Path = LEDGER) -> dict[str, Any]:
    reg = json.loads(REGISTRATION.read_text())
    candidate = Path(reg["frozen_record"]).parent
    problems = verify(reg, data, candidate)
    if problems:
        raise PermissionError("Prospective check void: " + "; ".join(problems))
    if command == "decide":
        if final_round is None:
            raise ValueError("decide needs --final-round")
        lines = ledger.read_text().splitlines() if ledger.exists() else []
        if any(json.loads(x)["command"] == "decide" for x in lines if x.strip()):
            raise PermissionError("The decision has already been taken")
    config = json.loads(protocol.CONFIG.read_text())
    u = protocol.load_universe(data, config)
    p = reg["prospective"]
    last = final_round if final_round is not None else 99
    u.splits["prospective"] = [
        k for k in u.races if k[0] == p["year"] and p["rounds_from"] <= k[1] <= last
    ]
    if command == "decide" and final_round is not None:
        # Refuse before logging anything: a premature decide must not consume the decision.
        rounds = pd.read_parquet(data, columns=["year", "round_number"]).query("year == 2026")
        ingested = {int(r) for r in rounds["round_number"]}
        missing = [r for r in range(p["rounds_from"], final_round + 1) if r not in ingested]
        if missing:
            raise ValueError(
                f"Rounds {missing} are not ingested yet; decide only after the final round"
            )
        if not u.splits["prospective"]:
            raise ValueError("No usable prospective race (all wet/red-flagged); nothing to decide")
    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    if not u.splits["prospective"]:
        _log({"command": command, "utc": stamp, "races": 0}, ledger)
        return {
            "command": command,
            "prospective_races": 0,
            "note": "No usable 2026 round >= 19 in the data yet.",
        }
    cand = protocol.load_candidate(candidate)
    new = protocol.score_split(u, cand, "prospective")
    out: dict[str, Any] = {
        "command": command,
        "utc": stamp,
        "prospective_races": len(u.splits["prospective"]),
        "prospective_only": protocol.summarize(new, "prospective", config),
    }
    if command == "decide":
        pooled = pd.concat([pd.read_csv(reg["test_rows"]["path"]), new], ignore_index=True)
        stats = protocol.paired(pooled, "ensemble")
        rule = reg["decision"]["promote_ensemble_to_default_if"]
        legal = float((pooled["ensemble"] >= 1e4).mean()) <= rule["illegal_share"]
        promote = legal and stats["ci95_s"][1] < rule["upper_ci95_ensemble_minus_planner_s_below"]
        out |= {"pooled_2026": stats, "legal": legal, "promote_ensemble": promote}
    else:
        out["note"] = "INTERIM LOOK: monitoring only, no decision may be taken from it."
    _log(
        {k: v for k, v in out.items() if k != "prospective_only"}
        | {"races": out["prospective_races"]},
        ledger,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=["status", "decide"])
    ap.add_argument("--final-round", type=int)
    ap.add_argument("--data", type=Path, default=Path("data/strategy_gold_2021_2026.parquet"))
    args = ap.parse_args()
    print(json.dumps(run(args.command, args.data, args.final_round), indent=2))


if __name__ == "__main__":
    main()
