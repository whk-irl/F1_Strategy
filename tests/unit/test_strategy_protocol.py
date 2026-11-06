"""Protocol guards: split integrity, bootstrap statistics, single-use test ledger."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ml.strategy import protocol as proto
from ml.strategy.data import lake_circuit_key


def test_splits_must_be_disjoint_and_chronological() -> None:
    proto.assert_disjoint({"train": [(2021, 1)], "val": [(2024, 1)], "test": [(2026, 1)]})
    with pytest.raises(ValueError, match="both"):
        proto.assert_disjoint({"train": [(2021, 1)], "val": [(2021, 1)], "test": []})
    with pytest.raises(ValueError, match="precede"):
        proto.assert_disjoint({"train": [(2025, 1)], "val": [(2024, 1)], "test": []})


def test_lake_circuit_key_uses_2024_reference_round() -> None:
    assert lake_circuit_key(2026, 1, 3) == "australia"
    assert lake_circuit_key(2021, 20, 23) == "qatar"
    assert lake_circuit_key(2026, 14, 0) == "2026-14"


def test_paired_bootstrap_is_race_clustered() -> None:
    rows = []
    for race in range(30):
        shift = -1.0 if race % 2 else 1.0  # race-level effect; cars within a race agree
        for car in range(10):
            rows.append(
                {"year": 2024, "round": race, "planner": 100.0, "x": 100.0 + shift + 0.01 * car}
            )
    out = proto.paired(pd.DataFrame(rows), "x")
    assert out["mean_s"] == pytest.approx(0.045, abs=1e-3)
    assert out["races"] == 30
    assert out["car_races"] == 300
    lo, hi = out["ci95_s"]
    assert lo < 0 < hi  # 30 independent races, not 300 cars, set the width
    assert hi - lo > 0.5


def test_resampled_timeline_matches_neutral_share() -> None:
    class R:
        n_laps = 2000
        neutral_lap = np.r_[np.zeros(2, bool), np.random.default_rng(0).random(1999) < 0.1]

    out = proto.resample_timeline(R(), np.random.default_rng(1))  # type: ignore[arg-type]
    assert out[:2].sum() == 0
    assert out[2:].mean() == pytest.approx(0.1, abs=0.03)


def _candidate(tmp: Path) -> tuple[Path, Path]:
    cand = tmp / "cand"
    (cand / "seed-1").mkdir(parents=True)
    (cand / "seed-1" / "policy.zip").write_bytes(b"weights")
    data = tmp / "data.parquet"
    data.write_bytes(b"data")
    return cand, data


def test_test_split_requires_freeze_and_is_single_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")
    monkeypatch.setattr(proto, "CONFIG", cfg)
    cand, data = _candidate(tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(PermissionError, match="freeze"):
        proto.open_test(cand, data, ledger)
    frozen = proto.freeze(cand, data)
    with pytest.raises(FileExistsError):
        proto.freeze(cand, data)
    assert proto.open_test(cand, data, ledger) == frozen["candidate_id"]
    assert json.loads(ledger.read_text())["candidate_id"] == frozen["candidate_id"]
    with pytest.raises(PermissionError, match="already used"):
        proto.open_test(cand, data, ledger)


def test_test_split_rejects_changed_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")
    monkeypatch.setattr(proto, "CONFIG", cfg)
    cand, data = _candidate(tmp_path)
    proto.freeze(cand, data)
    (cand / "seed-1" / "policy.zip").write_bytes(b"retrained")
    with pytest.raises(PermissionError, match="changed"):
        proto.open_test(cand, data, tmp_path / "ledger.jsonl")
