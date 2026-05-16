"""Sequence tire model training — TCN+GRU or PatchTST.

Usage::

    python -m ml.models.tire_degradation.train_sequence --model tcn_gru
    python -m ml.models.tire_degradation.train_sequence --model patch_tst
    make train-tire-tcn
    make train-tire-pst
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Annotated, Literal

import mlflow
import mlflow.pytorch
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import typer
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ml.models._loader import load_gold_seasons
from ml.models.tire_degradation.model_patch_tst import PatchTSTTireModel
from ml.models.tire_degradation.model_tcn_gru import TCNGRUTireModel
from ml.models.tire_degradation.sequence_dataset import (
    N_FEATURES,
    StintSequenceDataset,
    compute_r2,
    to_arch_config,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class SequenceTireConfig(BaseSettings):
    """Hyper-parameters and data settings for sequence tire model training."""

    model_config = SettingsConfigDict(
        env_prefix="PITWALL_TIRE_SEQ_",
        env_file=".env",
        extra="ignore",
    )

    model_type: Literal["tcn_gru", "patch_tst"] = Field(default="tcn_gru")
    training_seasons: list[int] = Field(default=[2022, 2023, 2024, 2025])
    val_rounds: list[int] = Field(default=[5, 10, 15, 20])
    seq_len: int = Field(default=10)
    batch_size: int = Field(default=512)
    epochs: int = Field(default=60)
    lr: float = Field(default=1e-3)
    weight_decay: float = Field(default=1e-4)
    patience: int = Field(default=10)

    # Shared architecture
    d_model: int = Field(default=64)
    dropout: float = Field(default=0.1)

    # TCN+GRU only
    n_tcn_layers: int = Field(default=4)
    gru_hidden: int = Field(default=64)
    kernel_size: int = Field(default=3)

    # PatchTST only
    patch_len: int = Field(default=5)
    nhead: int = Field(default=4)
    num_transformer_layers: int = Field(default=2)

    mlflow_tracking_uri: str = Field(
        default="mlruns",
        validation_alias="PITWALL_MLFLOW_TRACKING_URI",
    )
    mlflow_experiment: str = Field(default="tire-sequence-models")
    output_dir: str = Field(default="models_baked")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def build_model(
    config: SequenceTireConfig,
    n_features: int,
    seq_len: int,
) -> nn.Module:
    """Instantiate the chosen model architecture from config.

    Args:
        config: Training / architecture configuration.
        n_features: Number of input features per time step.
        seq_len: Input sequence length.

    Returns:
        Uninitialised (random-weight) ``nn.Module``.
    """
    if config.model_type == "tcn_gru":
        return TCNGRUTireModel(
            n_features=n_features,
            d_model=config.d_model,
            n_tcn_layers=config.n_tcn_layers,
            gru_hidden=config.gru_hidden,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )
    return PatchTSTTireModel(
        n_features=n_features,
        seq_len=seq_len,
        patch_len=config.patch_len,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_transformer_layers,
        dropout=config.dropout,
    )


def _arch_config_dict(
    config: SequenceTireConfig, n_features: int, seq_len: int
) -> dict[str, object]:
    """Build the architecture dict persisted alongside the model weights.

    Args:
        config: Training configuration.
        n_features: Number of input features per time step.
        seq_len: Input sequence length.

    Returns:
        Dict ready for ``json.dump``.
    """
    shared = to_arch_config(seq_len=seq_len, n_features=n_features, d_model=config.d_model)
    if config.model_type == "tcn_gru":
        shared.update(
            {
                "model_type": "tcn_gru",
                "n_tcn_layers": config.n_tcn_layers,
                "gru_hidden": config.gru_hidden,
                "kernel_size": config.kernel_size,
                "dropout": config.dropout,
            }
        )
    else:
        shared.update(
            {
                "model_type": "patch_tst",
                "patch_len": config.patch_len,
                "nhead": config.nhead,
                "num_transformer_layers": config.num_transformer_layers,
                "dropout": config.dropout,
            }
        )
    return shared


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float]:
    """Compute MAE and R² on a data loader.

    Args:
        model: Model in eval mode.
        loader: Validation data loader.
        device: Device to run inference on.

    Returns:
        Tuple of ``(mae, r2)``.
    """
    model.eval()
    all_preds: list[npt.NDArray[np.float32]] = []
    all_targets: list[npt.NDArray[np.float32]] = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        preds = model(x_batch).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y_batch.numpy())
    preds_arr = np.concatenate(all_preds)
    targets_arr = np.concatenate(all_targets)
    mae = float(np.mean(np.abs(preds_arr - targets_arr)))
    r2 = compute_r2(preds_arr, targets_arr)
    return mae, r2


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def train(config: SequenceTireConfig | None = None) -> None:
    """Train a sequence tire model and register artifacts in MLflow.

    Training steps:
    1. Load gold data via :func:`~ml.models._loader.load_gold_seasons`.
    2. Build train/val :class:`~ml.models.tire_degradation.sequence_dataset.StintSequenceDataset`.
    3. Train with AdamW + CosineAnnealingLR + HuberLoss.
    4. Early stopping on validation MAE; restore best weights.
    5. Save ``model.pt``, ``norm_stats.json``, ``config.json`` locally and in MLflow.

    Args:
        config: Training configuration (defaults to env-var / defaults when ``None``).
    """
    if config is None:
        config = SequenceTireConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training %s on %s", config.model_type, device)

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment)

    df = load_gold_seasons(config.training_seasons)

    train_ds = StintSequenceDataset(
        df,
        norm_stats=None,
        val_rounds=config.val_rounds,
        is_val=False,
        seq_len=config.seq_len,
    )
    val_ds = StintSequenceDataset(
        df,
        norm_stats=train_ds.norm_stats,
        val_rounds=config.val_rounds,
        is_val=True,
        seq_len=config.seq_len,
    )

    logger.info("Train samples: %d  Val samples: %d", len(train_ds), len(val_ds))

    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    n_features = N_FEATURES
    model = build_model(config, n_features, config.seq_len).to(device)

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    # HuberLoss is robust to the occasional outlier lap (VSC exit, slow zone)
    # that would distort MSE training.
    criterion = nn.HuberLoss(delta=1.0)

    best_val_mae = float("inf")
    best_state: dict[str, torch.Tensor] = {}
    epochs_no_improve = 0

    arch_dict = _arch_config_dict(config, n_features, config.seq_len)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "model_type": config.model_type,
                "training_seasons": config.training_seasons,
                "val_rounds": config.val_rounds,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "patience": config.patience,
                **{
                    k: v
                    for k, v in arch_dict.items()
                    if k not in {"n_features", "seq_len", "model_type"}
                },
            }
        )

        for epoch in range(1, config.epochs + 1):
            model.train()
            train_losses: list[float] = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                optimizer.zero_grad()
                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            scheduler.step()
            avg_train_loss = float(np.mean(train_losses))

            val_mae, val_r2 = _evaluate(model, val_loader, device)
            mlflow.log_metrics(
                {
                    "train_huber_loss": avg_train_loss,
                    "val_mae_s": val_mae,
                    "val_r2": val_r2,
                },
                step=epoch,
            )
            logger.info(
                "Epoch %d/%d  train_loss=%.4f  val_mae=%.4f  val_r2=%.4f",
                epoch,
                config.epochs,
                avg_train_loss,
                val_mae,
                val_r2,
            )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= config.patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, config.patience)
                    break

        # Restore best weights before saving
        if best_state:
            model.load_state_dict(best_state)

        val_mae_final, val_r2_final = _evaluate(model, val_loader, device)
        mlflow.log_metrics({"best_val_mae_s": val_mae_final, "best_val_r2": val_r2_final})
        logger.info("Final  val_mae=%.4f  val_r2=%.4f", val_mae_final, val_r2_final)

        # ---------------------------------------------------------------------------
        # Persist artifacts
        # ---------------------------------------------------------------------------
        out_dir = pathlib.Path(config.output_dir) / f"tire_{config.model_type}"
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), out_dir / "model.pt")
        train_ds.norm_stats.to_json(str(out_dir / "norm_stats.json"))
        with open(out_dir / "config.json", "w") as fh:
            json.dump(arch_dict, fh, indent=2)

        logger.info("Saved artifacts to %s", out_dir)

        # MLflow artifact upload is best-effort; the baked files above are the
        # canonical source for Streamlit Cloud / Docker deployment.
        try:
            mlflow.pytorch.log_model(model, artifact_path="model")
            mlflow.log_artifact(str(out_dir / "norm_stats.json"))
            mlflow.log_artifact(str(out_dir / "config.json"))
        except Exception as exc:
            logger.warning("MLflow artifact upload skipped: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    model: Annotated[
        str,
        typer.Option("--model", help="Model type: tcn_gru or patch_tst."),
    ] = "tcn_gru",
    seasons: Annotated[
        str,
        typer.Option("--seasons", help="Comma-separated seasons, e.g. 2022,2023,2024,2025."),
    ] = "2022,2023,2024,2025",
) -> None:
    """CLI entry point for sequence tire model training."""
    model_type: Literal["tcn_gru", "patch_tst"] = "patch_tst" if model == "patch_tst" else "tcn_gru"
    season_list = [int(s.strip()) for s in seasons.split(",")]
    train(
        SequenceTireConfig(
            model_type=model_type,
            training_seasons=season_list,
        )
    )


if __name__ == "__main__":
    app()
