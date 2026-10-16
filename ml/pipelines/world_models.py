"""Staged supervised world-model training, independent of hosted services."""

from __future__ import annotations

import copy
import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from services.ingestion.acquire import write_json
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from ml.models.safety_car.train import FEATURE_COLS as SC_FEATURES
from ml.models.safety_car.train import TARGET_COL as SC_TARGET
from ml.models.safety_car.train import _build_sc_label
from ml.models.tire_degradation.model_tcn_gru import TCNGRUTireModel
from ml.models.tire_degradation.sequence_dataset import (
    FEATURE_COLS as SEQUENCE_FEATURES,
)
from ml.models.tire_degradation.sequence_dataset import StintSequenceDataset
from ml.models.tire_degradation.train import FEATURE_COLS as TIRE_FEATURES
from ml.pipelines.config import TrainingConfig

LOGGER = logging.getLogger(__name__)


class BoosterPredictor:
    """Small inference adapter compatible with the simulator's pyfunc interface."""

    def __init__(self, booster: lgb.Booster) -> None:
        self.booster = booster

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict in the stored training feature order."""
        return np.asarray(self.booster.predict(frame[self.booster.feature_name()], num_threads=1))


def tire_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Use prior driver pace, never the lap being predicted, as a pace covariate."""
    out = df.sort_values(["year", "round_number", "driver_number", "lap_number"]).copy()
    keys = ["year", "round_number", "driver_number"]
    out["lap_delta_to_field_median_s"] = (
        out.groupby(keys)["lap_delta_to_field_median_s"]
        .transform(lambda values: values.shift(1).expanding(min_periods=1).median())
        .fillna(0.0)
    )
    mask = (
        out["is_accurate"].fillna(False)
        & out["track_status_encoded"].eq(0)
        & ~out["pit_in_this_lap"]
        & ~out["pit_out_this_lap"]
        & out["tyre_life_laps"].ge(2)
        & out["lap_time_delta_s"].notna()
    )
    out = out[mask].copy()
    out["is_fresh_tyre"] = out["is_fresh_tyre"].fillna(False).astype(int)
    return out.dropna(subset=TIRE_FEATURES)


def safety_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build one observation per race lap and exclude right-censored horizons."""
    out = _build_sc_label(df)
    keys = ["year", "round_number", "lap_number"]
    out["sc_laps_since_last"] = out["sc_laps_since_last"].fillna(50.0).clip(upper=50)
    agg = dict.fromkeys([*SC_FEATURES, "track_status_encoded"], "median")
    agg.pop("lap_number")
    agg[SC_TARGET] = "max"
    out = out.groupby(keys, as_index=False).agg(agg)
    last = out.groupby(["year", "round_number"])["lap_number"].transform("max")
    return out[out["track_status_encoded"].eq(0) & out["lap_number"].le(last - 3)].dropna(
        subset=SC_FEATURES
    )


def train_boosters(
    splits: dict[str, pd.DataFrame], config: TrainingConfig
) -> tuple[BoosterPredictor, BoosterPredictor, dict[str, Any]]:
    """Pretrain on older events and continue boosting on newer training events."""
    metrics: dict[str, Any] = {}
    trained = {}
    for name, builder, features, target, objective in (
        ("tire", tire_frame, TIRE_FEATURES, "lap_time_delta_s", "regression_l1"),
        ("sc", safety_frame, SC_FEATURES, SC_TARGET, "binary"),
    ):
        previous = None
        for stage, train_key, val_key in (
            ("pretrain", "pretrain", "pretrain_val"),
            ("finetune", "finetune", "finetune_val"),
        ):
            train, val = builder(splits[train_key]), builder(splits[val_key])
            if train.empty or val.empty:
                raise ValueError(f"Empty {name} {stage} dataset")
            if name == "sc" and train[target].nunique() < 2:
                raise ValueError("SC training needs both positive and negative labels")
            estimator = lgb.LGBMClassifier if name == "sc" else lgb.LGBMRegressor
            model = estimator(
                objective=objective,
                n_estimators=400 if stage == "pretrain" else 120,
                learning_rate=0.04 if stage == "pretrain" else 0.015,
                num_leaves=31,
                min_child_samples=40,
                n_jobs=4,
                verbosity=-1,
                random_state=config.seed,
                deterministic=True,
                force_col_wise=True,
            )
            model.fit(
                train[features],
                train[target],
                init_model=previous,
                eval_set=[(val[features], val[target])],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            previous = model.booster_
            path = config.output_dir / name / stage / "model.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            previous.save_model(str(path))
            predictor = BoosterPredictor(previous)
            pred = predictor.predict(val)
            key = f"{name}_{stage}"
            metrics[key] = {"training_rows": len(train), "validation_rows": len(val)}
            if name == "tire":
                metrics[key]["val_mae_s"] = float(mean_absolute_error(val[target], pred))
            else:
                metrics[key]["val_average_precision"] = float(
                    average_precision_score(val[target], pred)
                )
                metrics[key]["val_prevalence"] = float(val[target].mean())
                if val[target].nunique() == 2:
                    metrics[key]["val_roc_auc"] = float(roc_auc_score(val[target], pred))
            LOGGER.info("%s: %s", key, metrics[key])
        assert previous is not None
        trained[name] = BoosterPredictor(previous)
    return trained["tire"], trained["sc"], metrics


@torch.no_grad()
def sequence_metrics(
    model: nn.Module, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]], device: str
) -> dict[str, float]:
    """Evaluate next-lap MAE and a persistence forecast on the same windows."""
    model.eval()
    errors = []
    for x, y in loader:
        pred = model(x.to(device)).cpu()
        errors.extend((pred - y).abs().tolist())
    if not errors:
        raise ValueError("No contiguous sequence windows in evaluation split")
    return {"mae_s": float(np.mean(errors))}


def train_sequences(splits: dict[str, pd.DataFrame], config: TrainingConfig) -> dict[str, Any]:
    """Transfer TCN/GRU weights and fixed source normalization to newer seasons."""
    device = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    device = "cpu" if device == "auto" else device
    torch.manual_seed(config.seed)
    model = TCNGRUTireModel(
        len(SEQUENCE_FEATURES), d_model=64, n_tcn_layers=3, gru_hidden=64, dropout=0.1
    ).to(device)
    norm = None
    results = {}
    for stage, train_key, val_key, epochs in (
        ("pretrain", "pretrain", "pretrain_val", config.sequence_epochs),
        ("finetune", "finetune", "finetune_val", config.sequence_finetune_epochs),
    ):
        train = StintSequenceDataset(
            splits[train_key], norm_stats=norm, seq_len=config.sequence_length
        )
        norm = train.norm_stats
        val = StintSequenceDataset(splits[val_key], norm_stats=norm, seq_len=config.sequence_length)
        if not len(train) or not len(val):
            raise ValueError(f"No contiguous sequence windows for {stage}")
        generator = torch.Generator().manual_seed(config.seed)
        loader = DataLoader(train, batch_size=config.batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(val, batch_size=config.batch_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4 if stage == "pretrain" else 5e-5)
        best, stale, weights = float("inf"), 0, copy.deepcopy(model.state_dict())
        history = []
        for epoch in range(epochs):
            model.train()
            for x, y in loader:
                optimizer.zero_grad()
                loss = nn.functional.huber_loss(model(x.to(device)), y.to(device))
                loss.backward()  # type: ignore[no-untyped-call]
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            mae = sequence_metrics(model, val_loader, device)["mae_s"]
            history.append({"epoch": epoch + 1, "val_mae_s": mae})
            LOGGER.info("Sequence %s epoch %d: val MAE %.3fs", stage, epoch + 1, mae)
            if mae < best:
                best, stale, weights = mae, 0, copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= config.patience:
                break
        model.load_state_dict(weights)
        out = config.output_dir / "tire_tcn_gru" / stage
        out.mkdir(parents=True, exist_ok=True)
        torch.save(weights, out / "model.pt")
        norm.to_json(str(out / "norm_stats.json"))
        write_json(
            out / "config.json",
            {
                "model_type": "tcn_gru",
                "seq_len": config.sequence_length,
                "n_features": len(SEQUENCE_FEATURES),
                "d_model": 64,
                "n_tcn_layers": 3,
                "gru_hidden": 64,
                "kernel_size": 3,
                "dropout": 0.1,
            },
        )
        write_json(out / "history.json", history)
        results[stage] = {
            "val_mae_s": best,
            "training_windows": len(train),
            "validation_windows": len(val),
        }
    assert norm is not None
    test = StintSequenceDataset(splits["test"], norm_stats=norm, seq_len=config.sequence_length)
    results["test"] = sequence_metrics(
        model, DataLoader(test, batch_size=config.batch_size), device
    )
    target_idx = SEQUENCE_FEATURES.index("lap_time_delta_s")
    persistence_errors = [
        abs(float(x[-1, target_idx] * norm.std[target_idx] + norm.mean[target_idx]) - float(y))
        for x, y in (test[i] for i in range(len(test)))
    ]
    results["test"]["persistence_mae_s"] = float(np.mean(persistence_errors))
    return results
