"""Run acquisition-ready, reproducible pretraining through fine-tuning.

Usage: python -m ml.pipelines.train --config configs/training.local.json --resume
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from services.ingestion.acquire import file_sha256, write_json
from services.simulator.env import F1RaceEnv
from services.simulator.pit_time import DEFAULT_MODEL, load_pit_time_model
from services.simulator.validator import validate_simulator
from sklearn.metrics import average_precision_score, mean_absolute_error

from ml.models.tire_degradation.train import FEATURE_COLS as TIRE_FEATURES
from ml.pipelines.config import TrainingConfig
from ml.pipelines.data import load_dataset, split_events, split_manifest
from ml.pipelines.policy import dry_races, fit_policy, historical_actions
from ml.pipelines.report import paired_event_difference, write_report
from ml.pipelines.world_models import (
    BoosterPredictor,
    safety_frame,
    tire_frame,
    train_boosters,
    train_sequences,
)

LOGGER = logging.getLogger(__name__)


def validate_policy_simulator(
    df: pd.DataFrame, tire: Any, sc: Any, threshold: float
) -> dict[str, Any]:
    """Gate the actual RL environment using historical actions, without recorded-time replay."""
    scores = []
    for (year, rnd), race in dry_races(df).groupby(["year", "round_number"]):
        simulated, actual = [], []
        total = int(race["lap_number"].max())
        for driver, rows in race.groupby("driver_number"):
            rows = rows.sort_values("lap_number")
            if rows["lap_number"].max() < total or pd.isna(rows.iloc[-1]["position"]):
                continue
            inner = F1RaceEnv(race, str(driver), tire, sc, noise_std=0.0, replay_mode=False)
            inner.reset(seed=42)
            actions = historical_actions(rows)
            for lap in range(1, total + 1):
                inner.step(actions.get(lap, 0))
            simulated.append(inner._cumulative_time_s)
            actual.append(float(rows.iloc[-1]["position"]))
            inner.close()
        if len(actual) < 3:
            raise ValueError(f"Insufficient comparable finishers for gate {year}/{rnd}")
        rho = float(spearmanr(actual, simulated).statistic)
        if not np.isfinite(rho):
            raise ValueError(f"Undefined simulator correlation for {year}/{rnd}")
        scores.append(
            {
                "year": int(race["year"].iloc[0]),
                "round": int(race["round_number"].iloc[0]),
                "rho": rho,
                "drivers": len(actual),
            }
        )
        LOGGER.info("RL simulator gate %d/%02d: rho=%.3f", year, rnd, rho)
    mean = float(np.mean([row["rho"] for row in scores])) if scores else 0.0
    return {
        "mean_rho": mean,
        "passed": bool(scores and mean >= threshold),
        "races": scores,
        "scope": "dry events without red flags; drivers completing the lead-lap distance",
    }


def code_fingerprint() -> str:
    """Hash executable pipeline sources as well as the resolved experiment config."""
    digest = hashlib.sha256()
    for root in (Path("ml"), Path("services")):
        for path in sorted(root.rglob("*.py")):
            digest.update(str(path).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def run(config: TrainingConfig, *, resume: bool = False, through: str = "policy") -> dict[str, Any]:
    """Execute stages, fail closed on the simulator gate, and persist exact provenance."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    df, provenance = load_dataset(config)
    if not DEFAULT_MODEL.exists():
        raise FileNotFoundError(
            "Measured pit-time model missing; run `uv run python -m ml.pipelines.pit_time` first"
        )
    pit_model_sha256 = file_sha256(DEFAULT_MODEL)
    pit_model = load_pit_time_model()
    splits = split_events(df, config)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "config": config.model_dump(mode="json"),
                "provenance": provenance,
                "code": code_fingerprint(),
                "pit_time_model_sha256": pit_model_sha256,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    status_path = config.output_dir / "run.json"
    existing = json.loads(status_path.read_text()) if status_path.exists() else None
    if existing and existing["fingerprint"] != fingerprint:
        raise ValueError(
            "Run directory belongs to different code/config/data; use a new output_dir"
        )
    if existing and not resume:
        raise ValueError("Run directory already exists; use --resume or a new output_dir")
    status = existing or {
        "fingerprint": fingerprint,
        "started_at": datetime.now(UTC).isoformat(),
        "stages": {},
        "status": "running",
    }
    write_json(config.output_dir / "config.json", config.model_dump(mode="json"))
    write_json(config.output_dir / "dataset.json", provenance)
    write_json(
        config.output_dir / "pit_time_source.json",
        {"path": str(DEFAULT_MODEL.resolve()), "sha256": pit_model_sha256, "model": pit_model},
    )
    write_json(config.output_dir / "splits.json", split_manifest(splits))
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "fastf1",
            "pandas",
            "numpy",
            "torch",
            "lightgbm",
            "stable-baselines3",
            "pandera",
        )
    }
    write_json(
        config.output_dir / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": versions,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "code_sha256": code_fingerprint(),
        },
    )
    status["status"] = "running"
    write_json(status_path, status)

    def stage(name: str, action: Any) -> Any:
        previous = status["stages"].get(name)
        metric_path = config.output_dir / f"{name}.json"
        if resume and previous and previous["status"] == "complete":
            for filename, digest in previous["files"].items():
                if not Path(filename).exists() or file_sha256(Path(filename)) != digest:
                    raise ValueError(f"Corrupt or missing artifact: {filename}")
            LOGGER.info("Resuming: verified stage %s", name)
            return json.loads(metric_path.read_text())
        status["stages"][name] = {"status": "running"}
        write_json(status_path, status)
        LOGGER.info("Starting stage %s", name)
        try:
            value = action()
            write_json(metric_path, value)
            stage_dirs = {
                "world_models": ["tire", "sc"],
                "sequence": ["tire_tcn_gru"],
                "gate": [],
                "policy": ["policy"],
            }
            created = {
                p
                for folder in stage_dirs[name]
                for p in (config.output_dir / folder).rglob("*")
                if p.is_file()
            }
            if name == "gate":
                created.add(config.output_dir / "simulator_validation.json")
            created.add(metric_path)
            status["stages"][name] = {
                "status": "complete",
                "files": {str(path.resolve()): file_sha256(path) for path in sorted(created)},
            }
            write_json(status_path, status)
            return value
        except Exception as error:
            status["status"] = "failed"
            status["stages"][name] = {"status": "failed", "error": str(error)}
            write_json(status_path, status)
            raise

    if through == "prepare":
        status["status"] = "prepared"
        write_json(status_path, status)
        return status

    def world_stage() -> dict[str, Any]:
        _, _, metrics = train_boosters(splits, config)
        return metrics

    stage("world_models", world_stage)
    import lightgbm as lgb

    tire = BoosterPredictor(
        lgb.Booster(model_file=str(config.output_dir / "tire/finetune/model.txt"))
    )
    sc = BoosterPredictor(lgb.Booster(model_file=str(config.output_dir / "sc/finetune/model.txt")))
    if through == "world":
        return status
    stage("sequence", lambda: train_sequences(splits, config))
    if through == "sequence":
        return status

    def gate_stage() -> dict[str, Any]:
        legacy = validate_simulator(splits["gate"], tire, min_spearman=config.min_spearman)
        actual = validate_policy_simulator(splits["gate"], tire, sc, config.min_spearman)
        value = {"legacy_replay": legacy, "policy_environment": actual}
        write_json(config.output_dir / "simulator_validation.json", value)
        if not legacy["passed"] or not actual["passed"]:
            raise RuntimeError(
                "Simulator fidelity gate failed; policy training is blocked. "
                "See simulator_validation.json"
            )
        return value

    gate = stage("gate", gate_stage)
    if through == "gate":
        return status
    result = stage("policy", lambda: fit_policy(splits, tire, sc, config))
    tire_test, sc_test = tire_frame(splits["test"]), safety_frame(splits["test"])
    report = {
        "fingerprint": fingerprint,
        "algorithm": "behavioral cloning → 4-frame PPO → newer-event PPO fine-tuning",
        "policy": result["test"],
        "world_model_test": {
            "tire_mae_s": float(
                mean_absolute_error(
                    tire_test["lap_time_delta_s"], tire.predict(tire_test[TIRE_FEATURES])
                )
            ),
            "tire_zero_baseline_mae_s": float(tire_test["lap_time_delta_s"].abs().mean()),
            "sc_average_precision": float(
                average_precision_score(sc_test["is_sc_next_3_laps"], sc.predict(sc_test))
            ),
            "sc_prevalence": float(sc_test["is_sc_next_3_laps"].mean()),
        },
        "pit_time_model": {
            "sha256": pit_model_sha256,
            "train_scope": pit_model["train_scope"],
            "global_measured_lane_s": pit_model["global"]["lane_s"],
            "global_estimated_net_loss_s": pit_model["global"]["net_loss_s"],
        },
        "limitations": [
            "Retrospective recorded-field simulator, not a causal forecast of opponents",
            "Legacy timestamps approximate; feature values rebuilt causally",
            "Policy limited to dry races without red flags; no tire inventory model",
            "One training seed; paired episodes are not independent training runs",
            "Research candidate; historical success does not establish real-race superiority",
            "Pit race-loss labels are observational adjacent-lap estimates; "
            "disrupted stops are excluded",
        ],
    }
    fine = result["test"]["finetuned"]
    baseline = result["test"]["dynamic_programming"]
    report["candidate_passes_benchmark"] = (
        fine["mean_reward"] >= baseline["mean_reward"] and fine["compound_compliance"] == 1.0
    )
    report["finetuned_vs_dynamic_programming"] = paired_event_difference(
        fine, baseline, config.seed
    )
    write_json(config.output_dir / "report.json", report)
    write_report(config.output_dir, report, gate)
    write_json(
        config.output_dir / "manifest.json",
        {
            "format_version": 1,
            "algorithm": "bc_ppo_history",
            "frames": config.history_frames,
            "features_per_frame": 24,
            "feature_version": "causal-v3",
            "config": "config.json",
            "policy": "policy/finetuned.zip",
            "tire_model": "tire/finetune/model.txt",
            "sc_model": "sc/finetune/model.txt",
            "pit_time_model": "pit_time_source.json",
            "report": "report.json",
            "promoted": False,
            "sha256": {
                p.relative_to(config.output_dir).as_posix(): file_sha256(p)
                for p in config.output_dir.rglob("*")
                if p.is_file() and p.suffix in (".txt", ".zip", ".pt")
            },
        },
    )
    if config.tracking_uri:
        import mlflow

        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment("pitwall-staged-training")
        with mlflow.start_run(run_id=status.get("mlflow_run_id")) as tracked:
            status["mlflow_run_id"] = tracked.info.run_id
            mlflow.log_params(config.model_dump(mode="json"))
            mlflow.set_tag("fingerprint", fingerprint)
            for name, value in report["world_model_test"].items():
                mlflow.log_metric(name, value)
            for name, metrics in report["policy"].items():
                mlflow.log_metrics(
                    {
                        f"{name}_{key}": value
                        for key, value in metrics.items()
                        if isinstance(value, (int, float))
                    }
                )
            for name in (
                "config.json",
                "dataset.json",
                "pit_time_source.json",
                "splits.json",
                "report.md",
                "report.json",
                "environment.json",
                "simulator_validation.json",
                "sequence.json",
                "manifest.json",
            ):
                mlflow.log_artifact(str(config.output_dir / name))
            for name in ("policy", "tire", "sc", "tire_tcn_gru"):
                mlflow.log_artifacts(str(config.output_dir / name), artifact_path=name)
    status["status"] = "complete"
    status["finished_at"] = datetime.now(UTC).isoformat()
    write_json(status_path, status)
    return report


def main() -> None:
    """Run all stages or stop at an explicit inspection boundary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--through", choices=["prepare", "world", "sequence", "gate", "policy"], default="policy"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(TrainingConfig.load(args.config), resume=args.resume, through=args.through)


if __name__ == "__main__":
    main()
