"""Sequence tire model inference.

Implements two interfaces:

1. **Scalar** (backward-compatible with LightGBM ``predict.py``) — for the
   simulator's batch tire table.  History is synthesised by repeating the
   single input step SEQ_LEN times (steady-state approximation).
2. **Multi-step forecast** from real stint history — for Streamlit
   visualisation and offline evaluation.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn

_F32Array = npt.NDArray[np.float32]

from ml.models.tire_degradation.model_patch_tst import PatchTSTTireModel
from ml.models.tire_degradation.model_tcn_gru import TCNGRUTireModel
from ml.models.tire_degradation.sequence_dataset import (
    FEATURE_COLS,
    N_FEATURES,
    SEQ_LEN,
    NormStats,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
# Fallback for when the module is imported from a site-packages installation
# (Streamlit Cloud installs local packages via pip; CWD is the repo root there).
if not (_REPO_ROOT / "models_baked").exists():
    _REPO_ROOT = pathlib.Path.cwd()

# Re-export SequenceTireWrapper from the dedicated wrapper module so callers
# that use ``from predict_sequence import SequenceTireWrapper`` still work.
from ml.models.tire_degradation.wrapper import SequenceTireWrapper  # noqa: E402

__all__ = [
    "SequenceTireWrapper",
    "load_sequence_model",
    "predict_lap_time_delta",
    "predict_stint_from_history",
    "predict_stint_degradation",
]


def load_sequence_model(
    model_type: Literal["tcn_gru", "patch_tst"],
) -> tuple[nn.Module, NormStats, dict[str, Any]]:
    """Load a trained sequence model from the baked-artifact directory.

    Artifact layout expected under ``<repo_root>/models_baked/tire_{model_type}/``::

        model.pt          # state dict
        norm_stats.json   # NormStats serialised via NormStats.to_dict()
        config.json       # architecture hyper-parameters

    Args:
        model_type: Which architecture to load, either ``"tcn_gru"`` or
            ``"patch_tst"``.

    Returns:
        Tuple of ``(model, norm_stats, arch_config)`` where ``model`` is in
        eval mode on CPU.

    Raises:
        FileNotFoundError: If any of the three artifact files are missing.
        KeyError: If ``config.json`` is missing a required architecture field.
    """
    artifact_dir = _REPO_ROOT / "models_baked" / f"tire_{model_type}"

    with open(artifact_dir / "config.json") as fh:
        arch_config: dict[str, Any] = json.load(fh)

    norm_stats = NormStats.from_json(str(artifact_dir / "norm_stats.json"))

    model = _reconstruct_model(arch_config)
    state = torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    return model, norm_stats, arch_config


def _reconstruct_model(arch_config: dict[str, Any]) -> nn.Module:
    """Re-instantiate a model from a persisted architecture config dict.

    Args:
        arch_config: Dict produced by ``train_sequence._arch_config_dict``.

    Returns:
        Uninitialised ``nn.Module`` (weights not loaded yet).
    """
    model_type: str = arch_config["model_type"]
    if model_type == "tcn_gru":
        return TCNGRUTireModel(
            n_features=int(arch_config["n_features"]),
            d_model=int(arch_config["d_model"]),
            n_tcn_layers=int(arch_config["n_tcn_layers"]),
            gru_hidden=int(arch_config["gru_hidden"]),
            kernel_size=int(arch_config["kernel_size"]),
            dropout=float(arch_config["dropout"]),
        )
    return PatchTSTTireModel(
        n_features=int(arch_config["n_features"]),
        seq_len=int(arch_config["seq_len"]),
        patch_len=int(arch_config["patch_len"]),
        d_model=int(arch_config["d_model"]),
        nhead=int(arch_config["nhead"]),
        num_layers=int(arch_config["num_transformer_layers"]),
        dropout=float(arch_config["dropout"]),
    )


def predict_lap_time_delta(
    tyre_life_laps: int,
    compound_encoded: int,
    race_progress: float,
    track_status_encoded: int,
    is_fresh_tyre: bool,
    lap_delta_to_field_median_s: float,
    model: nn.Module,
    norm_stats: NormStats,
    sc_laps_since_last: float = 50.0,
    position: float = 10.0,
) -> float:
    """Predict the lap time delta for a single tyre state (scalar interface).

    This function is backward-compatible with the LightGBM ``predict.py``
    scalar interface.  Because a sequence model requires historical context,
    the single input step is **repeated** ``SEQ_LEN`` times to form a
    synthetic history — a steady-state approximation.

    Limitation: this approximation is accurate only when the tyre state has
    been stable for many laps.  For the first few laps of a stint, where
    rapid degradation is occurring, the autoregressive
    :func:`predict_stint_from_history` interface produces more reliable
    estimates.

    Args:
        tyre_life_laps: Number of laps on the current set of tyres.
        compound_encoded: 0=SOFT, 1=MEDIUM, 2=HARD, 3=INTER, 4=WET, 5=UNKNOWN.
        race_progress: Fraction of race completed (0.0–1.0).
        track_status_encoded: 0=clear, 1=yellow, 2=VSC, 3=SC, 4=red.
        is_fresh_tyre: Whether the tyre set is new.
        lap_delta_to_field_median_s: Car's typical pace vs field (seconds).
        model: Pre-loaded sequence model in eval mode.
        norm_stats: Normalisation statistics matching the model.
        sc_laps_since_last: Laps since the last safety car (default 50 = none).
        position: Current race position (default 10 = midfield).

    Returns:
        Predicted lap time delta in seconds relative to the driver's stint
        median.
    """
    step = np.array(
        [
            tyre_life_laps,
            compound_encoded,
            race_progress,
            track_status_encoded,
            int(is_fresh_tyre),
            lap_delta_to_field_median_s,
            0.0,  # lap_time_delta_s — unknown for current step; use 0.0 as neutral
            0.0,  # tyre_deg_rate_s_per_lap — ditto
            sc_laps_since_last,
            position,
        ],
        dtype=np.float32,
    )
    # Repeat to fill the window — steady-state approximation.
    window = np.tile(step, (SEQ_LEN, 1))  # (SEQ_LEN, N_FEATURES)
    normed = norm_stats.normalize(window)
    x = torch.from_numpy(normed).unsqueeze(0)  # (1, SEQ_LEN, N_FEATURES)
    with torch.no_grad():
        pred = model(x)
    return float(pred.item())


def predict_stint_from_history(
    stint_history: pd.DataFrame,
    n_forecast_laps: int,
    model: nn.Module,
    norm_stats: NormStats,
) -> _F32Array:
    """Autoregressively forecast lap-time deltas from real stint history.

    Each forecast step feeds its own prediction back as ``lap_time_delta_s``
    in the next input window, giving the model realistic context for
    subsequent predictions.

    Args:
        stint_history: DataFrame with columns matching :data:`FEATURE_COLS`,
            sorted ascending by lap number.  At least 1 row required.
        n_forecast_laps: Number of future laps to predict.
        model: Pre-loaded sequence model in eval mode.
        norm_stats: Normalisation statistics matching the model.

    Returns:
        Array of shape ``(n_forecast_laps,)`` with predicted lap-time deltas
        in seconds relative to the driver's stint median.
    """
    # Work on a copy so we don't mutate the caller's DataFrame.
    history = stint_history[FEATURE_COLS].copy().reset_index(drop=True)

    preds: list[float] = []

    for _ in range(n_forecast_laps):
        n_rows = len(history)

        if n_rows >= SEQ_LEN:
            window_df = history.iloc[-SEQ_LEN:]
        else:
            # Left-pad with zeros when real history is shorter than SEQ_LEN.
            pad = pd.DataFrame(
                np.zeros((SEQ_LEN - n_rows, N_FEATURES), dtype=np.float32),
                columns=FEATURE_COLS,
            )
            window_df = pd.concat([pad, history], ignore_index=True)

        window = window_df.to_numpy(dtype=np.float32)  # (SEQ_LEN, N_FEATURES)
        normed = norm_stats.normalize(window)
        x = torch.from_numpy(normed).unsqueeze(0)  # (1, SEQ_LEN, N_FEATURES)
        with torch.no_grad():
            delta = float(model(x).item())
        preds.append(delta)

        last_row = history.iloc[-1]
        last_delta = float(last_row["lap_time_delta_s"])

        next_row = pd.DataFrame(
            [
                {
                    "tyre_life_laps": float(last_row["tyre_life_laps"]) + 1.0,
                    "compound_encoded": float(last_row["compound_encoded"]),
                    # Advance race_progress by ~one lap fraction.
                    "race_progress": min(float(last_row["race_progress"]) + 0.02, 1.0),
                    "track_status_encoded": 0.0,
                    # After the first lap of a stint, the tyre is always worn.
                    "is_fresh_tyre": 0.0,
                    "lap_delta_to_field_median_s": float(last_row["lap_delta_to_field_median_s"]),
                    # Feed prediction back as context for the next window.
                    "lap_time_delta_s": delta,
                    # Degrade rate derived from successive predicted deltas.
                    "tyre_deg_rate_s_per_lap": delta - last_delta,
                    # SC counter increments each lap during green-flag running.
                    "sc_laps_since_last": min(float(last_row["sc_laps_since_last"]) + 1.0, 99.0),
                    # Position carried forward — not predictable in a pure stint forecast.
                    "position": float(last_row["position"]),
                }
            ]
        )
        history = pd.concat([history, next_row[FEATURE_COLS]], ignore_index=True)

    return np.array(preds, dtype=np.float32)


def predict_stint_degradation(
    stint_laps: int,
    compound_encoded: int,
    race_progress_start: float,
    lap_delta_to_field_median_s: float,
    model: nn.Module,
    norm_stats: NormStats,
) -> _F32Array:
    """Predict lap-time deltas for a projected stint (scalar-compatible interface).

    Builds a synthetic one-lap starting history (fresh tyre, tyre_life_laps=1)
    and delegates to :func:`predict_stint_from_history` for the autoregressive
    rollout.

    This mirrors the signature of the LightGBM ``predict_stint_degradation``
    but adds ``model`` and ``norm_stats`` parameters because a sequence model
    cannot be loaded from MLflow registry the same way.

    Args:
        stint_laps: Number of laps to project forward.
        compound_encoded: Compound integer code.
        race_progress_start: Race progress at stint start (0.0–1.0).
        lap_delta_to_field_median_s: Car baseline pace delta vs field (seconds).
        model: Pre-loaded sequence model in eval mode.
        norm_stats: Normalisation statistics matching the model.

    Returns:
        Array of predicted lap time deltas, one per lap in the stint.
    """
    seed_row = pd.DataFrame(
        [
            {
                "tyre_life_laps": 1.0,
                "compound_encoded": float(compound_encoded),
                "race_progress": race_progress_start,
                "track_status_encoded": 0.0,
                "is_fresh_tyre": 1.0,
                "lap_delta_to_field_median_s": lap_delta_to_field_median_s,
                "lap_time_delta_s": 0.0,
                "tyre_deg_rate_s_per_lap": 0.0,
                "sc_laps_since_last": 50.0,
                "position": 10.0,
            }
        ]
    )
    return predict_stint_from_history(
        seed_row[FEATURE_COLS],
        n_forecast_laps=stint_laps,
        model=model,
        norm_stats=norm_stats,
    )
