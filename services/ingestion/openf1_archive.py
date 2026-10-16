"""Historical OpenF1 fallback, retaining raw responses and conservative quality flags."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import requests

from services.ingestion.acquire import (
    FEATURE_VERSION,
    file_sha256,
    join_weather,
    write_json,
    write_parquet,
)
from services.ingestion.schemas.bronze import BronzeLapSchema
from services.ingestion.storage import build_parquet_key
from services.ingestion.transforms import bronze_to_silver, silver_to_gold


class ArchiveClient:
    """Rate-limited HTTP client with immutable session-scoped JSON caching."""

    def __init__(self, cache: Path) -> None:
        self.cache = cache
        self.session = requests.Session()
        self._last_request = 0.0

    def get(self, endpoint: str, **params: Any) -> list[dict[str, Any]]:
        """Fetch a documented public endpoint with bounded retries for transient errors."""
        key = "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
        path = self.cache / f"{endpoint}_{key}.json"
        if path.exists() and endpoint != "sessions":
            return cast(list[dict[str, Any]], json.loads(path.read_text(encoding="utf-8")))
        for attempt in range(4):
            time.sleep(max(0.0, 2.1 - (time.monotonic() - self._last_request)))
            self._last_request = time.monotonic()
            response = self.session.get(
                f"https://api.openf1.org/v1/{endpoint}", params=params, timeout=45
            )
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(min(float(response.headers.get("Retry-After", 5 * 2**attempt)), 60))
                continue
            response.raise_for_status()
            records = response.json()
            if not isinstance(records, list):
                raise ValueError(f"Unexpected {endpoint} response")
            write_json(path, records)
            return cast(list[dict[str, Any]], records)
        raise RuntimeError(f"Retries exhausted for {endpoint}")


def canonical_laps(records: dict[str, list[dict[str, Any]]], origin: pd.Timestamp) -> pd.DataFrame:
    """Translate archive responses to bronze, with weather/positions joined backward."""
    laps = pd.DataFrame(records["laps"])
    if laps.empty:
        raise ValueError("OpenF1 returned no laps")
    laps = laps.sort_values(["driver_number", "lap_number"]).reset_index(drop=True)
    out = pd.DataFrame(index=laps.index)
    out["DriverNumber"] = laps["driver_number"].astype(str)
    drivers = {str(row["driver_number"]): row for row in records["drivers"]}
    out["Driver"] = out["DriverNumber"].map(lambda value: drivers[value]["name_acronym"])
    out["Team"] = out["DriverNumber"].map(lambda value: drivers[value]["team_name"])
    out["LapNumber"] = laps["lap_number"].astype(int)
    start = pd.to_datetime(laps["date_start"], utc=True, format="mixed")
    out["LapStartTime"] = start - origin
    for src, dst in (
        ("lap_duration", "LapTime"),
        ("duration_sector_1", "Sector1Time"),
        ("duration_sector_2", "Sector2Time"),
        ("duration_sector_3", "Sector3Time"),
    ):
        out[dst] = pd.to_timedelta(pd.to_numeric(laps[src], errors="coerce"), unit="s")
    out["Time"] = out["LapStartTime"] + out["LapTime"]
    out["Stint"], out["TyreLife"] = np.nan, np.nan
    out["Compound"] = "UNKNOWN"
    out["FreshTyre"] = False
    for stint in records["stints"]:
        first, last = stint.get("lap_start"), stint.get("lap_end")
        if first is None:
            continue
        mask = out["DriverNumber"].eq(str(stint["driver_number"])) & out["LapNumber"].between(
            first, last if last is not None else int(out["LapNumber"].max())
        )
        out.loc[mask, "Stint"] = float(stint["stint_number"])
        out.loc[mask, "Compound"] = stint.get("compound") or "UNKNOWN"
        age = float(stint.get("tyre_age_at_start") or 0)
        out.loc[mask, "TyreLife"] = out.loc[mask, "LapNumber"] - first + age + 1
        out.loc[mask, "FreshTyre"] = age == 0
    out["PitInTime"] = pd.Series(pd.NaT, index=out.index, dtype="timedelta64[ns]")
    out["PitOutTime"] = pd.Series(pd.NaT, index=out.index, dtype="timedelta64[ns]")
    for pit in records["pit"]:
        mask = out["DriverNumber"].eq(str(pit["driver_number"])) & out["LapNumber"].eq(
            pit["lap_number"]
        )
        out.loc[mask, "PitInTime"] = pd.Timestamp(pit["date"]) - origin
    out.loc[laps["is_pit_out_lap"].fillna(False), "PitOutTime"] = out["LapStartTime"]
    out["Position"] = np.nan
    positions = pd.DataFrame(records["position"])
    if positions.empty:
        raise ValueError("OpenF1 positions are required for race training")
    positions["date"] = pd.to_datetime(positions["date"], utc=True, format="mixed")
    for driver, group in laps.groupby("driver_number"):
        known = positions[positions["driver_number"].eq(driver)].sort_values("date")
        times = (
            pd.DataFrame(
                {
                    "date": start.loc[group.index] + out.loc[group.index, "LapTime"],
                    "_row": group.index,
                }
            )
            .dropna(subset=["date"])
            .sort_values("date")
        )
        if known.empty or times.empty:
            continue
        joined = pd.merge_asof(times, known[["date", "position"]], on="date", direction="backward")
        out.loc[joined["_row"], "Position"] = joined["position"].to_numpy()
    # A lap receives all global statuses overlapping it; local yellows are conservative.
    changes = [(origin, "1")]
    for message in records["race_control"]:
        text = str(message.get("message", "")).upper()
        flag = message.get("flag")
        code = None
        if "VIRTUAL SAFETY CAR DEPLOYED" in text:
            code = "6"
        elif "SAFETY CAR DEPLOYED" in text:
            code = "4"
        elif flag == "RED":
            code = "5"
        elif "VSC ENDING" in text or "VIRTUAL SAFETY CAR ENDING" in text:
            code = "7"
        elif flag in ("GREEN", "CLEAR") or "TRACK CLEAR" in text or "DRS ENABLED" in text:
            code = "1"
        elif flag in ("YELLOW", "DOUBLE YELLOW") and message.get("scope") == "Track":
            code = "2"
        if code is not None:
            changes.append((pd.Timestamp(message["date"]), code))
    changes.sort(key=lambda item: item[0])
    out["TrackStatus"] = "1"
    for idx in out.index:
        begin = start.loc[idx]
        end = begin + out.loc[idx, "LapTime"]
        prior = [code for date, code in changes if date <= begin]
        during = [code for date, code in changes if begin < date <= end]
        out.loc[idx, "TrackStatus"] = "".join(sorted({prior[-1] if prior else "1", *during}))
    sectors = sum(pd.to_numeric(laps[f"duration_sector_{i}"], errors="coerce") for i in (1, 2, 3))
    duration = pd.to_numeric(laps["lap_duration"], errors="coerce")
    out["IsAccurate"] = duration.gt(0) & (sectors - duration).abs().lt(0.05)
    weather = pd.DataFrame(records["weather"])
    if not weather.empty:
        weather["Time"] = pd.to_datetime(weather["date"], utc=True, format="mixed") - origin
        weather = weather.rename(
            columns={
                "track_temperature": "TrackTemp",
                "air_temperature": "AirTemp",
                "rainfall": "Rainfall",
            }
        )
    return join_weather(out, weather)


def acquire_openf1(
    client: ArchiveClient, year: int, round_number: int, date: pd.Timestamp, root: Path
) -> dict[str, Any]:
    """Acquire a matched historical race and publish validated canonical layers."""
    sessions = client.get("sessions", year=year, session_name="Race")
    matches = [
        session
        for session in sessions
        if abs(pd.Timestamp(session["date_start"]) - date) < pd.Timedelta(days=2)
        and not session.get("is_cancelled", False)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one OpenF1 race for {year} round {round_number}; got {len(matches)}"
        )
    session = matches[0]
    records = {
        endpoint: client.get(endpoint, session_key=session["session_key"])
        for endpoint in ("laps", "drivers", "stints", "pit", "position", "race_control", "weather")
    }
    origin = pd.Timestamp(session["date_start"])
    raw = canonical_laps(records, origin)
    BronzeLapSchema.validate(raw)
    if raw.duplicated(["DriverNumber", "LapNumber"]).any():
        raise ValueError("Duplicate driver/lap keys in OpenF1")
    total = int(raw["LapNumber"].max())
    silver = bronze_to_silver(raw, year, round_number, "R")
    gold = silver_to_gold(silver, total, session_start=origin)
    files = {}
    for layer, frame in (("bronze", raw), ("silver", silver), ("gold", gold)):
        path = root / build_parquet_key(layer, year, round_number, "R")
        write_parquet(path, frame)
        files[layer] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    return {
        "status": "complete",
        "source": "OpenF1",
        "feature_version": FEATURE_VERSION,
        "year": year,
        "round_number": round_number,
        "session_key": session["session_key"],
        "total_laps": total,
        "distance_source": "observed_max_lap",
        "rows": len(raw),
        "quality_note": "Accuracy inferred from sector sum; deleted-lap status unavailable",
        "session_origin_utc": str(origin),
        "files": files,
    }
