"""Resumable local FastF1 acquisition with the same validated lake contracts.

Run ``python -m services.ingestion.acquire --years 2025,2026``. No storage or
tracking server is needed. Every successfully acquired session has a checksum
and provenance record; failed sessions remain visible and are retried next run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import fastf1
import numpy as np
import pandas as pd
from fastf1.exceptions import DataNotLoadedError
from pydantic import BaseModel, Field

from services.ingestion.schemas.bronze import BronzeLapSchema
from services.ingestion.storage import build_parquet_key
from services.ingestion.transforms import bronze_to_silver, silver_to_gold
from services.simulator.circuits import LOCATION_REFERENCE

LOGGER = logging.getLogger(__name__)
FEATURE_VERSION = "causal-v3"


class AcquisitionConfig(BaseModel):
    """Network acquisition settings, including an explicit historical cutoff."""

    years: list[int] = Field(default_factory=lambda: [2025, 2026])
    rounds: list[int] | None = None
    root: Path = Path("data/lake")
    cache: Path = Path(".fastf1_cache")
    as_of: datetime = Field(default_factory=lambda: datetime.now(UTC))
    retries: int = Field(default=3, ge=1, le=5)
    provider: Literal["auto", "fastf1", "openf1"] = "auto"
    retry_delay_s: float = Field(default=3.0, ge=0)


def file_sha256(path: Path) -> str:
    """Return a file digest without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomically replace a JSON record; interrupted writes never look complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Atomically publish one Parquet partition."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def completed_events(schedule: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
    """Select races at least one day before cutoff, avoiding in-progress races."""
    dates = pd.to_datetime(schedule["Session5DateUtc"], utc=True)
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise ValueError("as_of must include a timezone")
    return schedule[(schedule["RoundNumber"] > 0) & (dates + pd.Timedelta(days=1) <= cutoff)]


def join_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest weather known at lap start with a backward as-of join."""
    out = laps.copy()
    for column in ("track_temp_c", "air_temp_c", "rainfall"):
        out[column] = np.nan
    if weather.empty or "LapStartTime" not in out or "Time" not in weather:
        return out
    mapping = {"TrackTemp": "track_temp_c", "AirTemp": "air_temp_c", "Rainfall": "rainfall"}
    available = [column for column in mapping if column in weather]
    right = weather[["Time", *available]].dropna(subset=["Time"]).rename(columns=mapping)
    for column in available:
        right[mapping[column]] = pd.to_numeric(right[mapping[column]], errors="coerce").astype(
            float
        )
    left = pd.DataFrame({"_row": np.arange(len(out)), "Time": out["LapStartTime"].to_numpy()})
    left = left.dropna(subset=["Time"]).sort_values("Time")
    if left.empty or right.empty:
        return out
    joined = pd.merge_asof(left, right.sort_values("Time"), on="Time", direction="backward")
    for column in available:
        column_index = int(out.columns.get_indexer(pd.Index([mapping[column]]))[0])
        out.iloc[joined["_row"].astype(int).tolist(), column_index] = joined[
            mapping[column]
        ].to_numpy()
    return out


def session_origin(session: Any) -> pd.Timestamp:
    """Return a UTC race origin even when FastF1 did not load telemetry timing.

    Older FastF1 sessions do not expose ``t0_date`` after a laps-only load.  The
    published race start in the event schedule is the correct causal fallback
    and keeps historical acquisition independent of telemetry downloads.
    """
    try:
        value = session.t0_date
    except (AttributeError, DataNotLoadedError):
        value = session.event["Session5DateUtc"]
    origin = pd.Timestamp(value)
    return origin.tz_localize("UTC") if origin.tzinfo is None else origin.tz_convert("UTC")


def acquire_session(year: int, round_number: int, root: Path) -> dict[str, Any]:
    """Load laps, weather and metadata once, then validate and publish all layers."""
    session = fastf1.get_session(year, round_number, "R")
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    raw = pd.DataFrame(session.laps).reset_index(drop=True)
    if raw.empty:
        raise ValueError("Session has no lap data")
    if raw.duplicated(["DriverNumber", "LapNumber"]).any():
        raise ValueError("Duplicate driver/lap keys in source")
    total_laps = int(session.total_laps or raw["LapNumber"].max())
    origin = session_origin(session)
    raw = join_weather(raw, pd.DataFrame(session.weather_data))
    raw["IsAccurate"] = raw["IsAccurate"].fillna(False).astype(bool)
    BronzeLapSchema.validate(raw)
    silver = bronze_to_silver(raw, year=year, round_number=round_number, session="R")
    gold = silver_to_gold(silver, total_laps=total_laps, session_start=origin)
    paths = {}
    for layer, frame in (("bronze", raw), ("silver", silver), ("gold", gold)):
        path = root / build_parquet_key(layer, year, round_number, "R")
        write_parquet(path, frame)
        paths[layer] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    return {
        "status": "complete",
        "feature_version": FEATURE_VERSION,
        "year": year,
        "round_number": round_number,
        "session": "R",
        "event": str(session.event["EventName"]),
        "total_laps": total_laps,
        "session_origin_utc": str(origin),
        "rows": len(raw),
        "source": "FastF1",
        "fastf1_version": fastf1.__version__,
        "acquired_at": datetime.now(UTC).isoformat(),
        "files": paths,
    }


def acquire(config: AcquisitionConfig) -> dict[str, Any]:
    """Acquire completed sessions, verifying cached checksums before skipping them."""
    config.cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(config.cache))
    from services.ingestion.openf1_archive import ArchiveClient, acquire_openf1

    archive = ArchiveClient(config.root / "openf1_raw")
    manifest_path = config.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"sessions": {}}
    manifest["config"] = config.model_dump(mode="json")
    failures = []
    for year in config.years:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        events = completed_events(schedule, config.as_of)
        for _, event in events.iterrows():
            rnd = int(event["RoundNumber"])
            if config.rounds is not None and rnd not in config.rounds:
                continue
            key = f"{year}/{rnd:02d}/R"
            previous = manifest["sessions"].get(key, {})
            files = previous.get("files", {})
            if (
                previous.get("feature_version") == FEATURE_VERSION
                and len(files) == 3
                and all(
                    Path(item["path"]).exists()
                    and file_sha256(Path(item["path"])) == item["sha256"]
                    for item in files.values()
                )
            ):
                previous["circuit_reference_round"] = LOCATION_REFERENCE.get(
                    str(event["Location"]), 0
                )
                LOGGER.info("Verified cached session %s", key)
                continue
            for attempt in range(config.retries):
                try:
                    LOGGER.info("Acquiring %s (%s)", key, event["EventName"])
                    if config.provider == "openf1":
                        result = acquire_openf1(
                            archive,
                            year,
                            rnd,
                            pd.Timestamp(event["Session5DateUtc"], tz="UTC"),
                            config.root,
                        )
                    else:
                        try:
                            result = acquire_session(year, rnd, config.root)
                        except Exception:
                            if config.provider == "fastf1":
                                raise
                            LOGGER.warning("FastF1 unavailable for %s; trying OpenF1", key)
                            result = acquire_openf1(
                                archive,
                                year,
                                rnd,
                                pd.Timestamp(event["Session5DateUtc"], tz="UTC"),
                                config.root,
                            )
                    manifest["sessions"][key] = result
                    result["circuit_reference_round"] = LOCATION_REFERENCE.get(
                        str(event["Location"]), 0
                    )
                    break
                except Exception as error:
                    LOGGER.warning("%s attempt %d failed: %s", key, attempt + 1, error)
                    if attempt + 1 == config.retries:
                        failures.append(key)
                        manifest["sessions"][key] = {"status": "failed", "error": str(error)}
                    else:
                        time.sleep(config.retry_delay_s * 2**attempt)
            write_json(manifest_path, manifest)
    write_json(manifest_path, manifest)
    if failures:
        raise RuntimeError(f"Acquisition failed for {failures}; see {manifest_path}")
    return manifest


def main() -> None:
    """Parse acquisition options and run with a nonzero exit on incomplete data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", default="2025,2026")
    parser.add_argument("--rounds", default=None)
    parser.add_argument("--root", type=Path, default=Path("data/lake"))
    parser.add_argument("--as-of", default=datetime.now(UTC).isoformat())
    parser.add_argument("--provider", choices=["auto", "fastf1", "openf1"], default="auto")
    args = parser.parse_args()
    config = AcquisitionConfig(
        years=[int(x) for x in args.years.split(",")],
        root=args.root,
        rounds=[int(x) for x in args.rounds.split(",")] if args.rounds else None,
        as_of=datetime.fromisoformat(args.as_of),
        provider=args.provider,
    )
    logging.basicConfig(level=logging.INFO)
    acquire(config)


if __name__ == "__main__":
    main()
