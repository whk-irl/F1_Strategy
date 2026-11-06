"""Build the strategy dataset: 2022-25 gold file plus 2021/2026 lake partitions.

Each row gets a ``circuit_key``. 2022-25 use ``race_model.CALENDAR``; lake races
use the manifest's ``circuit_reference_round`` (a 2024 round number, 0 = venue
with no earlier reference), so no calendar is written by hand for 2021 or 2026.

    python -m ml.strategy.data --base data/gold_2022_2025.parquet \
        --lake data/lake --out data/strategy_gold_2021_2026.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml.strategy.race_model import CALENDAR, circuit_of

REQUIRED = [
    "year", "round_number", "session", "driver_number", "lap_number", "lap_time_s",
    "compound", "compound_encoded", "tyre_life_laps", "is_fresh_tyre", "position",
    "track_status", "pit_in_this_lap", "pit_out_this_lap", "is_accurate",
]  # fmt: skip


def lake_circuit_key(year: int, rnd: int, reference_round: int | None) -> str:
    """Circuit key from a manifest reference round on the 2024 calendar."""
    ref = int(reference_round or 0)
    if 1 <= ref <= len(CALENDAR[2024]):
        return CALENDAR[2024][ref - 1]
    return f"{year}-{rnd}"  # venue without an earlier reference: its own key


def load_lake(lake: Path, years: tuple[int, ...]) -> pd.DataFrame:
    sessions = json.loads((lake / "manifest.json").read_text())["sessions"]
    frames = []
    for key, meta in sorted(sessions.items()):
        y, r, s = key.split("/")
        if s != "R" or int(y) not in years or meta.get("status") != "complete":
            continue
        path = lake / "gold" / f"year={y}" / f"round={r}" / "session=R" / "laps.parquet"
        df = pd.read_parquet(path)
        df["year"], df["round_number"] = int(y), int(r)
        df["circuit_key"] = lake_circuit_key(int(y), int(r), meta.get("circuit_reference_round"))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build(base: Path, lake: Path, lake_years: tuple[int, ...] = (2021, 2026)) -> pd.DataFrame:
    old = pd.read_parquet(base)
    old["circuit_key"] = [
        circuit_of(y, r) for y, r in zip(old["year"], old["round_number"], strict=True)
    ]
    new = load_lake(lake, lake_years)
    overlap = set(map(tuple, old[["year", "round_number"]].drop_duplicates().to_numpy())) & set(
        map(tuple, new[["year", "round_number"]].drop_duplicates().to_numpy())
    )
    if overlap:
        raise ValueError(f"Races present in both sources: {sorted(overlap)}")
    cols = [*REQUIRED, "circuit_key"]
    missing = [c for c in cols if c not in new.columns]
    if missing:
        raise ValueError(f"Lake races lack required columns: {missing}")
    out = pd.concat([old[cols], new[cols]], ignore_index=True)
    out["is_accurate"] = out["is_accurate"].fillna(False).astype(bool)
    return out.sort_values(
        ["year", "round_number", "driver_number", "lap_number"], ignore_index=True
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=Path("data/gold_2022_2025.parquet"))
    ap.add_argument("--lake", type=Path, default=Path("data/lake"))
    ap.add_argument("--out", type=Path, default=Path("data/strategy_gold_2021_2026.parquet"))
    args = ap.parse_args()
    df = build(args.base, args.lake)
    df.to_parquet(args.out, index=False)
    races = df[["year", "round_number"]].drop_duplicates()
    print(f"{len(df)} laps, {len(races)} races -> {args.out}")
    print(races.groupby("year").size().to_dict())


if __name__ == "__main__":
    main()
