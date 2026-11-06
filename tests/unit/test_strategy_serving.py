"""Serving layer: advisor decisions, frozen-weight guard, API, prospective guards."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
from ml.strategy.race_model import TyreParams
from ml.strategy_serving import prospective
from ml.strategy_serving.advisor import CarState, StrategyAdvisor
from ml.strategy_serving.priors import GLOBAL

HIGH_WEAR = TyreParams(deg=(0.25, 0.20, 0.15), curv=(0.002, 0.002, 0.002), p_neutral=0.0)


def _priors(tmp: Path, params: TyreParams = HIGH_WEAR) -> Path:
    path = tmp / "priors.json"
    path.write_text(
        json.dumps(
            {"format_version": 1, "priors": {GLOBAL: asdict(params), "monza": asdict(params)}}
        )
    )
    return path


def _state(**kw: object) -> CarState:
    base = {
        "circuit": "monza",
        "total_laps": 53,
        "lap": 30,
        "compound": "MEDIUM",
        "tyre_age": 30,
        "compounds_used": ["MEDIUM"],
    }
    return CarState(**(base | kw))  # type: ignore[arg-type]


def test_old_high_wear_tyres_pit_and_fresh_ones_stay(tmp_path: Path) -> None:
    adv = StrategyAdvisor(_priors(tmp_path))
    old = adv.recommend(_state(), [])
    assert old.action.startswith("PIT")
    assert old.action != "PIT_MEDIUM"  # must add a second compound
    assert old.pit_saves_s > 0
    assert "saves" in old.explanation
    fresh = adv.recommend(_state(lap=10, tyre_age=3), [])
    assert fresh.action == "STAY"
    assert fresh.pit_saves_s < 0


def test_final_lap_never_pits_and_unknown_circuit_uses_global(tmp_path: Path) -> None:
    adv = StrategyAdvisor(_priors(tmp_path))
    last = adv.recommend(_state(lap=53, tyre_age=53, compounds_used=["MEDIUM", "HARD"]), [])
    assert last.action == "STAY"
    assert adv.recommend(_state(circuit="new-venue"), []).action.startswith("PIT")


def test_state_validation() -> None:
    with pytest.raises(ValueError, match="compounds_used"):
        _state(compounds_used=["HARD"])
    with pytest.raises(ValueError, match="total_laps"):
        _state(lap=60)


def _ensemble_dir(tmp: Path) -> Path:
    from ml.strategy.env import Scenario, StrategyEnv
    from stable_baselines3 import PPO

    root = tmp / "ens"
    policies = {}
    for seed in (1, 2):
        model = PPO(
            "MlpPolicy", StrategyEnv([Scenario(HIGH_WEAR, 20)]), n_steps=64, seed=seed, device="cpu"
        )
        (root / f"seed-{seed}").mkdir(parents=True)
        model.save(root / f"seed-{seed}" / "policy.zip")
        rel = f"seed-{seed}/policy.zip"
        policies[rel] = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    (root / "candidate.json").write_text(json.dumps({"observation": "v1"}))
    (root / "FROZEN.json").write_text(json.dumps({"candidate_id": "test123", "policies": policies}))
    return root


def test_ensemble_mode_serves_frozen_weights_only(tmp_path: Path) -> None:
    root = _ensemble_dir(tmp_path)
    adv = StrategyAdvisor(_priors(tmp_path), root)
    rec = adv.recommend(_state(), [], mode="ensemble")
    assert rec.mode == "ensemble"
    assert rec.ensemble_pit_probability is not None
    assert rec.model["ensemble_candidate"] == "test123"
    (root / "seed-1" / "policy.zip").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="frozen"):
        StrategyAdvisor(_priors(tmp_path), root)


def test_ensemble_mode_requires_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not configured"):
        StrategyAdvisor(_priors(tmp_path)).recommend(_state(), [], mode="ensemble")


def test_api_recommend_and_health(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient
    from services.api import strategy

    adv = StrategyAdvisor(_priors(tmp_path))
    monkeypatch.setattr(strategy, "advisor", lambda: adv)
    client = TestClient(strategy.app)
    assert client.get("/strategy/health").json()["modes"] == ["planner"]
    body = {"state": _state().model_dump(), "field_laps": []}
    r = client.post("/strategy/recommend", json=body)
    assert r.status_code == 200
    assert r.json()["action"].startswith("PIT")
    r = client.post("/strategy/recommend", json=body | {"mode": "ensemble"})
    assert r.status_code == 422


def test_history_hash_ignores_row_order_and_new_rounds_but_not_edits() -> None:
    df = pd.DataFrame(
        {
            "year": [2025, 2026, 2026],
            "round_number": [1, 18, 19],
            "driver_number": [1, 1, 1],
            "lap_number": [1, 1, 1],
            "lap_time_s": [90.0, 91.0, 92.0],
        }
    )
    h = prospective.history_sha(df)
    assert prospective.history_sha(df.iloc[::-1]) == h
    assert (
        prospective.history_sha(df.assign(lap_time_s=[90.0, 91.0, 99.0])) == h
    )  # round 19 is new data
    assert prospective.history_sha(df.assign(lap_time_s=[90.5, 91.0, 92.0])) != h


def test_decide_refuses_before_final_round_and_logs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ml.strategy import protocol

    monkeypatch.setattr(prospective, "verify", lambda *a: [])
    monkeypatch.setattr(prospective, "REGISTRATION", tmp_path / "reg.json")
    (tmp_path / "reg.json").write_text(
        json.dumps(
            {
                "frozen_record": str(tmp_path / "c" / "FROZEN.json"),
                "prospective": {"year": 2026, "rounds_from": 19},
            }
        )
    )
    monkeypatch.setattr(protocol, "load_universe", lambda d, c: protocol.Universe({}, {}, {}))
    monkeypatch.setattr(protocol, "CONFIG", tmp_path / "reg.json")
    data = tmp_path / "d.parquet"
    pd.DataFrame({"year": [2026], "round_number": [18]}).to_parquet(data)
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(ValueError, match="not ingested"):
        prospective.run("decide", data, 24, ledger)
    assert not ledger.exists()
