"""Fit robust pit-lane and effective race-loss estimates from recorded stops.

The race-loss target compares the in/out-lap pair with nearby clean laps for
the same driver. It is an observational estimate and may include tyre and
traffic effects. Event splits prevent later 2026 races influencing the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from services.ingestion.acquire import write_json
from services.simulator.circuits import LOCATION_REFERENCE
from services.simulator.pit_time import predict_pit_time
from sklearn.metrics import mean_absolute_error


def build_pit_frame(root: Path) -> pd.DataFrame:
    """Join cached OpenF1 pit records with laps without changing gold schemas."""
    manifest = json.loads((root / "pit_sessions.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for session in manifest["sessions"]:
        key = int(session["session_key"])
        venue = LOCATION_REFERENCE.get(session["circuit_short_name"], 0)
        pits = json.loads((root / "openf1_raw" / f"pit_session_key-{key}.json").read_text())
        laps = json.loads((root / "openf1_raw" / f"laps_session_key-{key}.json").read_text())
        by_driver: dict[int, dict[int, dict[str, Any]]] = {}
        for lap in laps:
            by_driver.setdefault(int(lap["driver_number"]), {})[int(lap["lap_number"])] = lap
        for pit in pits:
            lane = pit.get("lane_duration", pit.get("pit_duration"))
            if lane is None or not 12 <= float(lane) <= 45:
                continue  # Missing timing, red flags, repairs and long garage visits.
            driver = int(pit["driver_number"])
            lap_no = int(pit["lap_number"])
            driver_laps = by_driver.get(driver, {})
            # OpenF1's pit lap generally names the out lap. Verify the lap flag;
            # a shifted or missing record must never create a training target.
            out_no = (
                lap_no
                if driver_laps.get(lap_no, {}).get("is_pit_out_lap")
                else lap_no + 1
                if driver_laps.get(lap_no + 1, {}).get("is_pit_out_lap")
                else None
            )
            net: float | None = None
            if out_no is not None:
                durations = {
                    number: float(lap["lap_duration"])
                    for number, lap in driver_laps.items()
                    if lap.get("lap_duration") is not None
                    and 45 <= float(lap["lap_duration"]) <= 180
                }
                before = [durations[n] for n in range(out_no - 5, out_no - 1) if n in durations]
                after = [durations[n] for n in range(out_no + 1, out_no + 5) if n in durations]
                if (
                    out_no - 1 in durations
                    and out_no in durations
                    and len(before) >= 2
                    and len(after) >= 2
                ):
                    candidate = (
                        durations[out_no - 1]
                        + durations[out_no]
                        - float(np.median(before))
                        - float(np.median(after))
                    )
                    # Adjacent-lap deltas can be dominated by safety cars,
                    # traffic, or red flags. Use only clean-looking stops to
                    # learn a nominal green-flag cost; preserve lane timing for
                    # every plausible stop regardless of this filter.
                    if 5 <= candidate <= 75 and abs(candidate - float(lane)) <= 8:
                        net = candidate
            rows.append(
                {
                    "year": int(session["year"]),
                    "round_number": int(session["round_number"]),
                    "session_key": key,
                    "reference_round": venue,
                    "driver_number": driver,
                    "date": str(pit.get("date", "")),
                    "lane_s": float(lane),
                    "net_loss_s": net,
                }
            )
    return pd.DataFrame(rows)


def _distribution(
    frame: pd.DataFrame, global_values: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Shrink venue medians and observed uncertainty toward the global fit."""
    n = len(frame)
    lane = frame["lane_s"].to_numpy(dtype=float)
    net = frame["net_loss_s"].dropna().to_numpy(dtype=float)
    if not n or not len(net):
        raise ValueError("Pit-time fit requires measured lane and net-loss targets")

    def values(data: np.ndarray, prefix: str) -> dict[str, float]:
        median, low, high = np.quantile(data, [0.5, 0.1, 0.9])
        if global_values is not None:
            weight = len(data) / (len(data) + 12)
            interval_prefix = "net" if prefix == "net_loss" else prefix
            median = weight * median + (1 - weight) * global_values[f"{prefix}_s"]
            low = weight * low + (1 - weight) * global_values[f"{interval_prefix}_low_s"]
            high = weight * high + (1 - weight) * global_values[f"{interval_prefix}_high_s"]
        return {
            f"{prefix}_s": round(float(median), 3),
            f"{prefix}_low_s": round(float(min(low, median)), 3),
            f"{prefix}_high_s": round(float(max(high, median)), 3),
        }

    net_values = values(net, "net_loss")
    return {
        "sample_count": n,
        "net_sample_count": len(net),
        **values(lane, "lane"),
        "net_loss_s": net_values["net_loss_s"],
        "net_low_s": net_values["net_loss_low_s"],
        "net_high_s": net_values["net_loss_high_s"],
    }


def fit_pit_model(frame: pd.DataFrame) -> dict[str, Any]:
    """Fit a venue-aware empirical predictor with a measured global fallback."""
    global_values = _distribution(frame)
    venues = {
        str(int(str(venue))): _distribution(group, global_values)
        for venue, group in frame.groupby("reference_round")
        if venue and group["net_loss_s"].notna().sum() >= 5
    }
    return {
        "format_version": 1,
        "target": "OpenF1 lane_duration plus observed adjacent-lap net race loss",
        "train_scope": "2023 and 2025 races plus 2026 rounds 1–10",
        "global": global_values,
        "venues": venues,
    }


def score_pit_model(model: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    clean = frame.dropna(subset=["net_loss_s"])
    if frame.empty:
        return {"rows": 0, "lane_mae_s": None, "net_mae_s": None}
    lane_forecasts = [predict_pit_time(model, int(v)) for v in frame["reference_round"].tolist()]
    net_forecasts = [predict_pit_time(model, int(v)) for v in clean["reference_round"].tolist()]
    online_errors: list[float] = []
    for _, race in frame.groupby("session_key"):
        prior: list[float] = []
        for row in race.sort_values("date").itertuples():
            base = predict_pit_time(model, int(str(row.reference_round)))["lane_s"]
            prior_sorted = sorted(prior)
            current = prior_sorted[len(prior_sorted) // 2] if prior_sorted else base
            predicted = base + len(prior) / (len(prior) + 8) * (current - base)
            online_errors.append(abs(float(str(row.lane_s)) - predicted))
            prior.append(float(str(row.lane_s)))
    return {
        "lane_rows": len(frame),
        "net_rows": len(clean),
        "events": int(frame["session_key"].nunique()),
        "lane_mae_s": round(
            float(mean_absolute_error(frame["lane_s"], [p["lane_s"] for p in lane_forecasts])),
            3,
        ),
        "lane_online_mae_s": round(float(np.mean(online_errors)), 3),
        "net_mae_s": round(
            float(
                mean_absolute_error(clean["net_loss_s"], [p["net_loss_s"] for p in net_forecasts])
            ),
            3,
        )
        if not clean.empty
        else None,
        "lane_global_mae_s": round(
            float(
                mean_absolute_error(frame["lane_s"], np.full(len(frame), model["global"]["lane_s"]))
            ),
            3,
        ),
        "net_global_mae_s": round(
            float(
                mean_absolute_error(
                    clean["net_loss_s"], np.full(len(clean), model["global"]["net_loss_s"])
                )
            ),
            3,
        )
        if not clean.empty
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/lake"))
    parser.add_argument("--output", type=Path, default=Path("models_baked/pit_time.json"))
    parser.add_argument("--report", type=Path, default=Path("runs/pit_time/report.json"))
    args = parser.parse_args()
    frame = build_pit_frame(args.root)
    if frame.empty:
        raise ValueError("No usable pit data; run scripts/acquire_pit_history.py")
    train = frame[
        (frame.year.isin([2023, 2025])) | ((frame.year == 2026) & (frame.round_number <= 10))
    ]
    gate = frame[frame.year.eq(2024)]
    val = frame[frame.year.eq(2026) & frame.round_number.between(11, 13)]
    test = frame[frame.year.eq(2026) & frame.round_number.ge(14)]
    model = fit_pit_model(train)
    report = {
        "train_rows": len(train),
        "train_net_rows": int(train.net_loss_s.notna().sum()),
        "gate_2024": score_pit_model(model, gate),
        "validation_2026_rounds_11_13": score_pit_model(model, val),
        "test_2026_rounds_14_plus": score_pit_model(model, test),
        "target_quality": (
            "Lane time uses 12–45 s routine measured stops. Nominal net loss uses nearby laps "
            "and excludes disrupted stops outside lane time ±8 s."
        ),
    }
    write_json(args.output, model)
    write_json(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
