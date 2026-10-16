"""Offline tests for archive mapping, retries and durable acquisition manifests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import Mock

import pandas as pd
import pytest
import requests
from services.ingestion import acquire as acquisition
from services.ingestion.acquire import AcquisitionConfig, acquire, file_sha256, write_parquet
from services.ingestion.openf1_archive import ArchiveClient, canonical_laps
from services.ingestion.transforms import bronze_to_silver, silver_to_gold


def test_archive_cache_and_transient_retry(tmp_path, monkeypatch):
    monkeypatch.setattr("services.ingestion.openf1_archive.time.sleep", lambda _: None)
    client = ArchiveClient(tmp_path)
    busy, success = Mock(), Mock()
    busy.status_code, busy.headers = 429, {"Retry-After": "0"}
    success.status_code, success.json.return_value = 200, [{"lap_number": 1}]
    client.session.get = Mock(side_effect=[busy, success])
    assert client.get("laps", session_key=42) == [{"lap_number": 1}]
    assert client.get("laps", session_key=42) == [{"lap_number": 1}]
    assert client.session.get.call_count == 2


def test_archive_does_not_cache_authentication_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("services.ingestion.openf1_archive.time.sleep", lambda _: None)
    client = ArchiveClient(tmp_path)
    response = Mock(status_code=401)
    response.raise_for_status.side_effect = requests.HTTPError("401")
    client.session.get = Mock(return_value=response)
    with pytest.raises(requests.HTTPError):
        client.get("laps", session_key=42)
    assert not list(tmp_path.glob("*.json"))


def test_openf1_mapping_validates_all_layers():
    origin = pd.Timestamp("2026-03-08T04:00:00Z")
    records = {
        "laps": [
            {
                "driver_number": 44,
                "lap_number": i + 1,
                "date_start": str(origin + pd.Timedelta(seconds=i * 90)),
                "lap_duration": 90.0,
                "duration_sector_1": 30.0,
                "duration_sector_2": 30.0,
                "duration_sector_3": 30.0,
                "is_pit_out_lap": i == 2,
            }
            for i in range(3)
        ],
        "drivers": [{"driver_number": 44, "name_acronym": "HAM", "team_name": "Ferrari"}],
        "stints": [
            {
                "driver_number": 44,
                "stint_number": 1,
                "compound": "MEDIUM",
                "lap_start": 1,
                "lap_end": 2,
                "tyre_age_at_start": 0,
            },
            {
                "driver_number": 44,
                "stint_number": 2,
                "compound": "HARD",
                "lap_start": 3,
                "lap_end": 3,
                "tyre_age_at_start": 2,
            },
        ],
        "pit": [
            {"driver_number": 44, "lap_number": 2, "date": str(origin + pd.Timedelta(seconds=175))}
        ],
        "position": [
            {"driver_number": 44, "position": 2, "date": str(origin + pd.Timedelta(seconds=1))}
        ],
        "race_control": [
            {
                "message": "SAFETY CAR DEPLOYED",
                "flag": None,
                "date": str(origin + pd.Timedelta(seconds=120)),
            }
        ],
        "weather": [
            {
                "date": str(origin + pd.Timedelta(seconds=5)),
                "track_temperature": 35,
                "air_temperature": 25,
                "rainfall": 0,
            }
        ],
    }
    raw = canonical_laps(records, origin)
    assert raw["DriverNumber"].eq("44").all()
    assert raw["IsAccurate"].all()
    assert raw["TyreLife"].tolist() == [1, 2, 3]
    assert pd.isna(raw.iloc[0]["track_temp_c"])
    assert raw.iloc[1]["track_temp_c"] == 35
    assert raw.iloc[1]["TrackStatus"] == "14"
    silver = bronze_to_silver(raw, 2026, 1, "R")
    assert silver["pit_in_this_lap"].tolist() == [False, True, False]
    assert silver["pit_out_this_lap"].tolist() == [False, False, True]
    gold = silver_to_gold(silver, 3, session_start=origin)
    assert gold.iloc[-1]["event_timestamp"] == origin + pd.Timedelta(seconds=270)

    # Real OpenF1 stops commonly name the out lap, which must label the
    # preceding lap as the pit decision for behavioral cloning.
    records["pit"][0]["lap_number"] = 3
    aligned = bronze_to_silver(canonical_laps(records, origin), 2026, 1, "R")
    assert aligned["pit_in_this_lap"].tolist() == [False, True, False]
    assert aligned["pit_out_this_lap"].tolist() == [False, False, True]


@pytest.fixture
def local_acquisition(tmp_path, monkeypatch):
    schedule = pd.DataFrame(
        {
            "RoundNumber": [1],
            "EventName": ["Australian Grand Prix"],
            "Location": ["Melbourne"],
            "Session5DateUtc": [pd.Timestamp("2026-03-08")],
        }
    )
    monkeypatch.setattr(acquisition.fastf1, "get_event_schedule", lambda *args, **kwargs: schedule)
    monkeypatch.setattr(acquisition.fastf1.Cache, "enable_cache", lambda *args: None)
    config = AcquisitionConfig(
        years=[2026],
        root=tmp_path / "lake",
        cache=tmp_path / "cache",
        as_of=datetime(2026, 10, 15, tzinfo=UTC),
        retries=1,
    )
    return config


def test_acquisition_resume_verifies_and_repairs_checksums(local_acquisition, monkeypatch):
    config = local_acquisition
    calls = []

    def fetch(year, rnd, root):
        calls.append((year, rnd))
        files = {}
        for layer in ("bronze", "silver", "gold"):
            path = root / layer / "laps.parquet"
            write_parquet(path, pd.DataFrame({"lap": [1]}))
            files[layer] = {"path": str(path), "sha256": file_sha256(path)}
        return {
            "status": "complete",
            "feature_version": acquisition.FEATURE_VERSION,
            "source": "FastF1",
            "files": files,
        }

    monkeypatch.setattr(acquisition, "acquire_session", fetch)
    acquire(config)
    acquire(config)
    assert len(calls) == 1
    (config.root / "gold/laps.parquet").write_bytes(b"incomplete")
    acquire(config)
    assert len(calls) == 2


def test_acquisition_failure_is_persisted(local_acquisition, monkeypatch):
    config = local_acquisition.model_copy(update={"provider": "fastf1"})
    monkeypatch.setattr(acquisition, "acquire_session", Mock(side_effect=RuntimeError("offline")))
    with pytest.raises(RuntimeError, match="Acquisition failed"):
        acquire(config)
    manifest = json.loads((config.root / "manifest.json").read_text())
    assert manifest["sessions"]["2026/01/R"]["status"] == "failed"


def test_auto_provider_records_fallback(local_acquisition, monkeypatch):
    monkeypatch.setattr(acquisition, "acquire_session", Mock(side_effect=RuntimeError("401")))
    fallback = Mock(return_value={"status": "complete", "source": "OpenF1"})
    monkeypatch.setattr("services.ingestion.openf1_archive.acquire_openf1", fallback)
    result = acquire(local_acquisition)
    assert result["sessions"]["2026/01/R"]["source"] == "OpenF1"
    assert fallback.call_count == 1
