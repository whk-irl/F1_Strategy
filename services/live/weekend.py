"""Acquire pre-race sessions and publish a measured weekend context.

Run ``python -m services.live.weekend --config configs/weekends/us-2026.json``.
Practice and qualifying stay outside the race-policy training lake. Only finished
pre-race sessions are read; the forthcoming race is never used for calibration.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from services.ingestion.acquire import file_sha256, write_json, write_parquet
from services.ingestion.openf1_archive import canonical_laps
from services.ingestion.schemas.bronze import BronzeLapSchema
from services.ingestion.transforms import bronze_to_silver, silver_to_gold
from services.live.race_data import ROOT, RaceFeed, RaceFeedError, load_bootstrap_track, timestamp

DEFAULT_CONFIG = ROOT / "configs/weekends/us-2026.json"
SESSION_CODES = {"Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3", "Qualifying": "Q"}
ENDPOINTS = ("laps", "drivers", "stints", "pit", "position", "race_control", "weather")


class WeekendConfig(BaseModel):
    """An explicit event identity and verified scheduled distance."""

    model_config = ConfigDict(extra="forbid")
    year: int
    round_number: int
    meeting_key: int
    race_session_key: int
    country_name: str
    circuit_short_name: str
    circuit_reference_round: int
    scheduled_laps: int = Field(ge=1, le=100)
    race_start_utc: datetime
    output_dir: Path
    snapshot_path: Path
    sources: list[str]


def practice_calibration(frame: pd.DataFrame) -> dict[str, Any]:
    """Aggregate within-stint slopes without pooling driver/session intercepts.

    Observed practice pace mixes fuel burn, traffic and tyre wear. Positive
    plausible slopes are shrunk toward the existing Austin prior; negative slopes
    do not become negative degradation. Cliff ages remain explicit priors.
    """
    defaults = {0: 0.092, 1: 0.0575, 2: 0.0345}
    names = {0: "SOFT", 1: "MEDIUM", 2: "HARD"}
    rates, evidence = {}, {}
    for compound in range(3):
        clean = frame[
            frame["session"].isin(["FP1", "FP2", "FP3"])
            & frame["compound_encoded"].eq(compound)
            & frame["is_accurate"].fillna(False)
            & frame["track_status_encoded"].eq(0)
            & ~frame["pit_in_this_lap"]
            & ~frame["pit_out_this_lap"]
            & frame["tyre_life_laps"].ge(3)
        ]
        slopes, laps, drivers = [], 0, set()
        for (_, driver, _), stint in clean.groupby(["session", "driver_number", "stint_number"]):
            stint = stint.dropna(subset=["lap_time_s", "tyre_life_laps"])
            if len(stint) < 6 or stint["tyre_life_laps"].nunique() < 5:
                continue
            # Trim obvious slow/cooldown laps relative to this stint, not fleet pace.
            median = float(stint["lap_time_s"].median())
            stint = stint[stint["lap_time_s"].between(median - 3, median + 3)]
            if len(stint) < 6:
                continue
            x = stint["tyre_life_laps"].to_numpy(dtype=float)
            y = stint["lap_time_s"].to_numpy(dtype=float)
            pair_slopes = [
                (y[j] - y[i]) / (x[j] - x[i])
                for i in range(len(x))
                for j in range(i + 1, len(x))
                if x[j] != x[i]
            ]
            slope = float(np.median(pair_slopes))
            if 0 < slope < 0.5:
                slopes.append(slope)
                laps += len(stint)
                drivers.add(str(driver))
        observed = float(np.median(slopes)) if slopes else None
        weight = min(0.7, len(slopes) / (len(slopes) + 5))
        rates[str(compound)] = (
            round((1 - weight) * defaults[compound] + weight * observed, 5)
            if observed is not None
            else defaults[compound]
        )
        evidence[names[compound]] = {
            "clean_laps": laps,
            "usable_stints": len(slopes),
            "drivers": sorted(drivers),
            "observed_slope_s_per_lap": observed,
            "prior_weight": round(1 - weight, 3),
            "source": "practice-with-prior" if slopes else "Austin circuit prior",
        }
    return {
        "degrade_rate": rates,
        "cliff_lap": {"0": 22, "1": 32, "2": 42},
        "compound_evidence": evidence,
        "note": "Within-stint practice pace is confounded by fuel and traffic; cliffs are priors.",
    }


def _classification(
    results: list[dict[str, Any]], drivers: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    roster = {int(row["driver_number"]): row for row in drivers}
    return [
        {
            "position": row.get("position"),
            "driver_number": row["driver_number"],
            "name_acronym": roster.get(int(row["driver_number"]), {}).get("name_acronym"),
            "team_name": roster.get(int(row["driver_number"]), {}).get("team_name"),
            "duration": row.get("duration", row.get("lap_duration")),
            "number_of_laps": row.get("number_of_laps"),
        }
        for row in sorted(results, key=lambda item: item.get("position") or 999)
    ]


def refresh_weekend(
    config: WeekendConfig,
    feed: RaceFeed,
    *,
    as_of: datetime | None = None,
    include_track: bool = False,
) -> dict[str, Any]:
    """Publish source records, validated partitions and an event-matched summary."""
    cutoff = timestamp(as_of or datetime.now(UTC))
    sessions = feed.get("sessions", {"meeting_key": config.meeting_key}, refresh=True)
    matching = [
        row
        for row in sessions
        if row.get("year") == config.year
        and row.get("circuit_short_name") == config.circuit_short_name
        and not row.get("is_cancelled", False)
    ]
    race = next((row for row in matching if row["session_key"] == config.race_session_key), None)
    if not race or race["session_name"] != "Race":
        raise ValueError("Configured race session does not match provider event metadata")
    if timestamp(race["date_start"]) != timestamp(config.race_start_utc):
        raise ValueError("Race start changed; verify official schedule and update config")
    acquired: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for meta in sorted(matching, key=lambda item: item["date_start"]):
        code = SESSION_CODES.get(meta["session_name"])
        if not code or timestamp(meta["date_end"]) > cutoff:
            continue
        folder = config.output_dir / code
        try:
            records = {
                endpoint: feed.get(endpoint, {"session_key": meta["session_key"]}, archive=True)
                for endpoint in ENDPOINTS
            }
            result = feed.get("session_result", {"session_key": meta["session_key"]}, refresh=True)
            if not result:
                raise ValueError("Session classification is not yet published")
            raw_path = folder / "source.json"
            write_json(raw_path, {"meta": meta, "records": records, "results": result})
            raw = canonical_laps(records, timestamp(meta["date_start"]))
            BronzeLapSchema.validate(raw)
            silver = bronze_to_silver(raw, config.year, config.round_number, code)
            gold = silver_to_gold(
                silver, config.scheduled_laps, session_start=timestamp(meta["date_start"])
            )
            gold["circuit_reference_round"] = config.circuit_reference_round
            gold_path = folder / "laps.parquet"
            write_parquet(gold_path, gold)
            frames.append(gold)
            acquired.append(
                {
                    "code": code,
                    "meta": meta,
                    "rows": len(gold),
                    "classification": _classification(result, records["drivers"]),
                    "files": {
                        "source": {"path": str(raw_path), "sha256": file_sha256(raw_path)},
                        "gold": {"path": str(gold_path), "sha256": file_sha256(gold_path)},
                    },
                }
            )
        except (RaceFeedError, ValueError) as error:
            failures.append({"code": code, "error": str(error)})
    if not frames:
        raise ValueError("No completed weekend sessions could be acquired")
    all_laps = pd.concat(frames, ignore_index=True)
    qualifying: list[dict[str, Any]] = next(
        (s["classification"] for s in acquired if s["code"] == "Q"), []
    )
    grid = []
    grid_file: dict[str, str] | None = None
    try:
        # OpenF1 attaches the race grid to the qualifying session, not the race key.
        # Meeting identity also distinguishes Austin from Miami/Las Vegas.
        rows = feed.get("starting_grid", {"meeting_key": config.meeting_key}, refresh=True)
        if rows:
            grid_path = config.output_dir / "starting_grid.source.json"
            write_json(grid_path, rows)
            grid_file = {"path": str(grid_path), "sha256": file_sha256(grid_path)}
        roster = [
            {
                "driver_number": row["driver_number"],
                "name_acronym": row["name_acronym"],
                "team_name": row["team_name"],
            }
            for row in qualifying
        ]
        grid = _classification(rows, roster)
    except RaceFeedError:
        pass  # Qualifying order remains explicitly provisional until grid is published.
    value = {
        "format_version": 1,
        "updated_at": cutoff.isoformat(),
        "year": config.year,
        "meeting_key": config.meeting_key,
        "circuit_short_name": config.circuit_short_name,
        "scheduled_laps": config.scheduled_laps,
        "race_session": race,
        "sessions": acquired,
        "qualifying": qualifying,
        "starting_grid": grid,
        "starting_grid_file": grid_file,
        "grid_status": "published starting grid" if grid else "qualifying order; grid pending",
        "calibration": practice_calibration(all_laps),
        "rows": len(all_laps),
        "failures": failures,
        "sources": config.sources,
        "status": "complete" if len(acquired) == 4 and not failures else "partial",
        "cutoff_note": (
            "as_of selects finished sessions; classifications reflect fetch-time revisions."
        ),
    }
    if include_track:
        track = load_bootstrap_track(feed, race, value)
        if track:
            track_path = config.output_dir / "track.source.json"
            write_json(track_path, track)
            value["track"] = track
            value["track_file"] = {
                "path": str(track_path),
                "sha256": file_sha256(track_path),
                "source": "OpenF1 completed practice/qualifying GPS at this meeting",
            }
    write_json(config.output_dir / "summary.json", value)
    write_json(config.snapshot_path, value)
    return value


def load_weekend_summary(meta: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Read local generated context or the shipped snapshot for the matching race."""
    if not DEFAULT_CONFIG.exists():
        return None
    try:
        config = WeekendConfig.model_validate_json(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        local = ROOT / config.output_dir / "summary.json"
        snapshot = ROOT / config.snapshot_path
        path = local if local.exists() else snapshot
        if not path.exists():
            return None
        value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if meta is not None and (
            meta.get("session_name") != "Race"
            or meta.get("meeting_key") != value["meeting_key"]
            or meta.get("session_key") != value["race_session"]["session_key"]
        ):
            return None
        return value
    except (OSError, ValueError, KeyError):
        return None


def main() -> None:
    """Refresh completed practice/qualifying and print only aggregate status."""
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = WeekendConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    result = refresh_weekend(config, RaceFeed(), include_track=True)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "rows", "grid_status", "failures", "calibration")
            },
            indent=2,
        )
    )
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
