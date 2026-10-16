"""Versioned feature rebuilds and event-disjoint dataset partitioning."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from services.ingestion.acquire import FEATURE_VERSION, file_sha256
from services.ingestion.schemas.gold import GoldLapSchema
from services.ingestion.schemas.silver import SilverLapSchema
from services.ingestion.transforms import silver_to_gold
from services.simulator.circuits import legacy_reference

from ml.pipelines.config import TrainingConfig


def load_dataset(config: TrainingConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild legacy gold causally, then replace whole sessions with fresh partitions."""
    frames = []
    provenance: dict[str, Any] = {"feature_version": FEATURE_VERSION, "sources": []}
    if config.gold_path.exists():
        legacy = pd.read_parquet(config.gold_path)
        for _, race in legacy.groupby(["year", "round_number", "session"]):
            race = race.copy()
            for col in ("lap_end_time_s", "track_temp_c", "air_temp_c", "rainfall"):
                race[col] = np.nan
            silver = SilverLapSchema.validate(race[list(SilverLapSchema.to_schema().columns)])
            frames.append(silver_to_gold(silver, int(race["lap_number"].max())))
        provenance["sources"].append(
            {
                "path": str(config.gold_path.resolve()),
                "sha256": file_sha256(config.gold_path),
                "timestamp_source": "legacy_synthetic",
                "features_recomputed": True,
            }
        )
    manifest_path = config.lake_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"sessions": {}}
    for path in sorted((config.lake_root / "gold").glob("year=*/round=*/session=R/laps.parquet")):
        fresh = GoldLapSchema.validate(pd.read_parquet(path))
        if (
            fresh.empty
            or fresh[["year", "round_number", "session"]].drop_duplicates().shape[0] != 1
        ):
            raise ValueError(f"Expected one nonempty session in {path}")
        key = tuple(fresh[["year", "round_number", "session"]].iloc[0])
        record = manifest["sessions"].get(f"{key[0]}/{key[1]:02d}/{key[2]}", {})
        if record.get("status") != "complete" or record.get("feature_version") != FEATURE_VERSION:
            raise ValueError(f"Partition lacks a completed current-version manifest: {path}")
        if record["files"]["gold"]["sha256"] != file_sha256(path):
            raise ValueError(f"Gold partition checksum mismatch: {path}")
        fresh["circuit_reference_round"] = record.get("circuit_reference_round", 0)
        frames = [
            frame
            for frame in frames
            if tuple(frame[["year", "round_number", "session"]].iloc[0]) != key
        ]
        frames.append(fresh)
        provenance["sources"].append(
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "timestamp_source": f"{record['source']}_UTC",
            }
        )
    if not frames:
        raise ValueError("No gold data found; run acquisition first")
    df = pd.concat(frames, ignore_index=True)
    if "circuit_reference_round" not in df:
        df["circuit_reference_round"] = np.nan
    df["circuit_reference_round"] = (
        df["circuit_reference_round"]
        .fillna(
            pd.Series(
                [
                    legacy_reference(int(y), int(r))
                    for y, r in zip(df["year"], df["round_number"], strict=True)
                ],
                index=df.index,
            )
        )
        .astype(int)
    )
    keys = ["year", "round_number", "session", "driver_number", "lap_number"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate session/driver/lap keys")
    return df.sort_values(keys).reset_index(drop=True), provenance


def split_events(df: pd.DataFrame, config: TrainingConfig) -> dict[str, pd.DataFrame]:
    """Reserve complete events, with a completely separate simulator gate season."""
    df = df[df["session"].eq("R")]
    val = df["round_number"].isin(config.validation_rounds)
    source = df["year"].isin(config.pretrain_years)
    recent = df["year"].isin(config.finetune_years)
    order = df["year"] * 100 + df["round_number"]
    val_start = config.finetune_validation_start[0] * 100 + config.finetune_validation_start[1]
    test_start = config.test_start[0] * 100 + config.test_start[1]
    masks = {
        "pretrain": source & ~val,
        "pretrain_val": source & val,
        "finetune": recent & (order < val_start),
        "finetune_val": recent & (order >= val_start) & (order < test_start),
        "test": recent & (order >= test_start),
        "gate": df["year"].isin(config.gate_years),
    }
    splits = {name: df[mask].copy() for name, mask in masks.items()}
    for name, frame in splits.items():
        if frame.empty:
            raise ValueError(f"Split {name!r} is empty; acquire data or revise explicit splits")
    sets = {
        name: set(map(tuple, frame[["year", "round_number"]].drop_duplicates().to_numpy()))
        for name, frame in splits.items()
    }
    for name, keys in sets.items():
        for other, other_keys in sets.items():
            if name != other and keys & other_keys:
                raise ValueError(f"Event leakage between {name} and {other}")
    return splits


def split_manifest(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Return the exact event membership and row counts for an experiment."""
    return {
        name: {
            "rows": len(frame),
            "events": frame[["year", "round_number"]]
            .drop_duplicates()
            .astype(int)
            .to_dict("records"),
        }
        for name, frame in splits.items()
    }
