"""Replay a recorded race: the advisor's call for one car on every lap.

The car follows its REAL strategy (advice is not applied), so each line shows what
the model would have said at that moment next to what the team actually did.

    python -m ml.strategy_serving.replay --year 2024 --round 16 --driver 16
    python -m ml.strategy_serving.replay --year 2024 --round 16 --driver 16 --mode ensemble
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import pandas as pd

from ml.strategy_serving.advisor import COMPOUNDS, CarState, FieldLap, StrategyAdvisor

NAMES = {0: "SOFT", 1: "MEDIUM", 2: "HARD", 3: "INTERMEDIATE", 4: "WET"}


def field_laps(race: pd.DataFrame) -> list[FieldLap]:
    def opt(v: object, cast: type) -> object:
        return None if pd.isna(v) else cast(v)

    return [
        FieldLap(
            driver_number=str(x.driver_number),
            lap_number=int(x.lap_number),
            lap_time_s=opt(x.lap_time_s, float),
            compound=NAMES[int(x.compound_encoded)],
            tyre_life_laps=opt(x.tyre_life_laps, int),
            track_status=str(x.track_status),
            pit_in=bool(x.pit_in_this_lap),
            pit_out=bool(x.pit_out_this_lap),
            is_accurate=bool(x.is_accurate),
        )
        for x in race.itertuples()
    ]


def replay(
    data: pd.DataFrame,
    advisor: StrategyAdvisor,
    year: int,
    rnd: int,
    driver: str,
    mode: Literal["planner", "ensemble"],
) -> pd.DataFrame:
    race = data[(data["year"] == year) & (data["round_number"] == rnd)]
    if race.empty:
        raise SystemExit(f"No race {year} round {rnd} in the dataset")
    car = race[race["driver_number"].astype(str) == driver].sort_values("lap_number")
    if car.empty:
        raise SystemExit(f"Driver {driver} not in {year} round {rnd}")
    circuit, total = str(race["circuit_key"].iloc[0]), int(race["lap_number"].max())
    laps = field_laps(race)
    flagged = race["track_status"].astype(str).str.contains("[467]")
    neutral_share = race.assign(n=flagged).groupby("lap_number")["n"].mean()
    used: list[str] = []
    rows = []
    for x in car.itertuples():
        comp = NAMES[int(x.compound_encoded)]
        if comp not in COMPOUNDS:
            rows.append({"lap": int(x.lap_number), "compound": comp, "advice": "wet: not covered"})
            continue
        if comp not in used:
            used.append(comp)
        lap = int(x.lap_number)
        state = CarState(
            circuit=circuit,
            total_laps=total,
            lap=lap,
            compound=comp,
            tyre_age=max(1, int(x.tyre_life_laps)) if pd.notna(x.tyre_life_laps) else 1,
            compounds_used=used,
            neutral=bool(neutral_share.get(lap, 0.0) >= 0.3),
        )
        rec = advisor.recommend(state, [f for f in laps if f.lap_number <= lap], mode)
        rows.append(
            {
                "lap": lap,
                "compound": comp,
                "age": state.tyre_age,
                "sc": "SC" if state.neutral else "",
                "team": "PIT" if bool(x.pit_in_this_lap) else "",
                "advice": rec.action,
                "pit_saves_s": rec.pit_saves_s,
                "p_pit": rec.ensemble_pit_probability,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--driver", required=True)
    ap.add_argument("--mode", choices=["planner", "ensemble"], default="planner")
    ap.add_argument("--data", type=Path, default=Path("data/strategy_gold_2021_2026.parquet"))
    ap.add_argument("--config", type=Path, default=Path("configs/strategy-serving.json"))
    args = ap.parse_args()
    advisor = StrategyAdvisor.from_config(args.config)
    out = replay(pd.read_parquet(args.data), advisor, args.year, args.round, args.driver, args.mode)
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
