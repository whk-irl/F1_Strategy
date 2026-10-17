"""Run the full world-model, simulator-gate, and temporal-IQL lifecycle.

Usage:
    python -m ml.pipelines.train_iql --config configs/training.iql-hpc.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
from services.ingestion.acquire import file_sha256, write_json

from ml.pipelines.config import TrainingConfig
from ml.pipelines.data import load_dataset, split_events
from ml.pipelines.iql import fit_temporal_iql
from ml.pipelines.report import paired_event_difference
from ml.pipelines.train import run
from ml.pipelines.world_models import BoosterPredictor

LOGGER = logging.getLogger(__name__)


def execute(config: TrainingConfig, *, resume: bool = False) -> dict[str, Any]:
    """Train causal world models, pass the replay gate, and fit temporal IQL."""
    tire_path = config.output_dir / "tire/finetune/model.txt"
    sc_path = config.output_dir / "sc/finetune/model.txt"
    gate_path = config.output_dir / "simulator_validation.json"
    if not (tire_path.exists() and sc_path.exists() and gate_path.exists()):
        run(config, resume=resume, through="gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate["legacy_replay"]["passed"] or not gate["policy_environment"]["passed"]:
        raise RuntimeError("Simulator gate did not pass; temporal IQL training is blocked")
    frame, provenance = load_dataset(config)
    splits = split_events(frame, config)
    tire = BoosterPredictor(lgb.Booster(model_file=str(tire_path)))
    sc = BoosterPredictor(lgb.Booster(model_file=str(sc_path)))
    result = fit_temporal_iql(splits, tire, sc, config)
    iql, baseline = result["test"]["iql"], result["test"]["dynamic_programming"]
    comparison = paired_event_difference(iql, baseline, config.seed)
    promoted = bool(
        iql["compound_compliance"] == 1.0
        and comparison["mean_event_time_difference_s"] <= 0
        and comparison["bootstrap_95_ci_s"][1] <= 0
    )
    manifest = {
        "format_version": 1,
        "algorithm": "temporal_iql",
        "frames": config.history_frames,
        "features_per_frame": 24,
        "policy": "policy_iql/finetuned.pt",
        "dataset_sources": provenance["sources"],
        "test_vs_dynamic_programming": comparison,
        "promoted": promoted,
        "sha256": {
            "policy_iql/finetuned.pt": file_sha256(config.output_dir / "policy_iql/finetuned.pt")
        },
    }
    write_json(config.output_dir / "iql_manifest.json", manifest)
    LOGGER.info("Temporal IQL candidate promotion status: %s", promoted)
    return {"metrics": result, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = TrainingConfig.load(args.config)
    overrides = {
        key: value
        for key, value in {"seed": args.seed, "output_dir": args.output_dir}.items()
        if value is not None
    }
    execute(config.model_copy(update=overrides), resume=args.resume)


if __name__ == "__main__":
    main()
