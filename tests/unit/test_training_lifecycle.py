"""Lifecycle safety and portable artifact regression tests, with training stubbed."""

from __future__ import annotations

import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from gymnasium import spaces
from ml.models.strategy_policy import predict_staged
from ml.pipelines import train as pipeline
from ml.pipelines.config import TrainingConfig
from ml.pipelines.report import paired_event_difference
from services.ingestion.acquire import file_sha256, write_json
from services.ingestion.transforms import silver_to_gold


@pytest.fixture
def stubbed_pipeline(tmp_path, monkeypatch, silver_laps_df):
    gold = silver_to_gold(silver_laps_df, 10)
    frames = [
        gold.assign(year=year, round_number=rnd)
        for year, rnd in [(2022, 1), (2022, 5), (2024, 1), (2025, 1), (2026, 11), (2026, 14)]
    ]
    monkeypatch.setattr(pipeline, "load_dataset", lambda config: (pd.concat(frames), {}))
    monkeypatch.setattr(pipeline, "train_boosters", lambda *args: (None, None, {}))
    monkeypatch.setattr(pipeline, "train_sequences", lambda *args: {})
    monkeypatch.setattr("lightgbm.Booster", lambda **kwargs: Mock())
    config = TrainingConfig(output_dir=tmp_path / "run", tracking_uri=None)
    return config


def test_failed_gate_blocks_policy_training(stubbed_pipeline, monkeypatch):
    config = stubbed_pipeline
    monkeypatch.setattr(pipeline, "validate_simulator", lambda *args, **kwargs: {"passed": False})
    monkeypatch.setattr(pipeline, "validate_policy_simulator", lambda *args: {"passed": True})
    fit = Mock()
    monkeypatch.setattr(pipeline, "fit_policy", fit)
    with pytest.raises(RuntimeError, match="gate failed"):
        pipeline.run(config)
    fit.assert_not_called()
    status = json.loads((config.output_dir / "run.json").read_text())
    assert status["status"] == "failed"
    assert status["stages"]["gate"]["status"] == "failed"


def test_resume_rejects_corrupted_stage_artifact(stubbed_pipeline):
    config = stubbed_pipeline
    pipeline.run(config, through="world")
    (config.output_dir / "world_models.json").write_text("{} ")
    with pytest.raises(ValueError, match="Corrupt or missing"):
        pipeline.run(config, resume=True, through="world")


def test_windows_manifest_keys_load_portably(tmp_path, monkeypatch):
    checkpoint = tmp_path / "policy/model.zip"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint fixture")
    manifest = {
        "format_version": 1,
        "features_per_frame": 24,
        "frames": 4,
        "policy": "policy/model.zip",
        "sha256": {"policy\\model.zip": file_sha256(checkpoint)},
        "promoted": False,
    }
    write_json(tmp_path / "manifest.json", manifest)
    fake = Mock()
    fake.observation_space = spaces.Box(-np.inf, np.inf, shape=(96,), dtype=np.float32)
    fake.predict.return_value = (np.array(2), None)
    monkeypatch.setattr(predict_staged.PPO, "load", lambda *args, **kwargs: fake)
    predictor = predict_staged.StagedPolicyPredictor(tmp_path)
    recommendation = predictor.recommend(np.zeros(21, dtype=np.float32), {0})
    assert recommendation["recommendation"] == "pit_medium"
    assert fake.predict.call_args[0][0].shape == (96,)
    predictor.reset()
    assert len(predictor.history) == 3
    checkpoint.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        predict_staged.StagedPolicyPredictor(tmp_path)


def test_uncertainty_resamples_events_not_driver_rows():
    rows = [
        {
            "year": 2026,
            "round": rnd,
            "driver": str(driver),
            "seed": driver,
            "time_s": float(driver + rnd),
        }
        for rnd in (14, 15)
        for driver in (1, 2, 3)
    ]
    baseline = {"episodes": rows}
    candidate = {"episodes": [{**row, "time_s": row["time_s"] - 2} for row in rows]}
    result = paired_event_difference(candidate, baseline, 42)
    assert result["events"] == 2
    assert result["bootstrap_95_ci_s"] == [-2.0, -2.0]


def test_unpaired_comparison_is_rejected():
    candidate = {"episodes": [{"year": 2026, "round": 14, "driver": "1", "seed": 42}]}
    baseline = {"episodes": [{"year": 2026, "round": 15, "driver": "1", "seed": 42}]}
    with pytest.raises(ValueError, match="not paired"):
        paired_event_difference(candidate, baseline, 42)
