"""Shared dataset and normalization utilities for sequence tire models.

Both the TCN+GRU and PatchTST models consume fixed-length sliding windows of
per-lap features from each driver stint.  This module centralises the data
preparation so the two model files stay architecture-only.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import Dataset

_F32Array = npt.NDArray[np.float32]

SEQ_LEN: int = 15

FEATURE_COLS: list[str] = [
    "tyre_life_laps",
    "compound_encoded",
    "race_progress",
    "track_status_encoded",
    "is_fresh_tyre",
    "lap_delta_to_field_median_s",
    "lap_time_delta_s",
    "tyre_deg_rate_s_per_lap",
    # Additional context features
    "sc_laps_since_last",   # track-state recovery signal after restarts
    "position",             # clean-air vs traffic context
]

TARGET_COL: str = "lap_time_delta_s"

N_FEATURES: int = len(FEATURE_COLS)


@dataclasses.dataclass
class NormStats:
    """Per-feature mean and standard deviation for z-score normalisation.

    Args:
        mean: Shape ``(N_FEATURES,)`` float32 array of feature means.
        std: Shape ``(N_FEATURES,)`` float32 array of feature standard deviations.
            Any zero-std feature is clamped to 1.0 to avoid division-by-zero.
    """

    mean: _F32Array
    std: _F32Array

    def __post_init__(self) -> None:
        self.std = np.where(self.std == 0, 1.0, self.std).astype(np.float32)
        self.mean = self.mean.astype(np.float32)

    def normalize(self, x: _F32Array) -> _F32Array:
        """Z-score normalise a feature array.

        Args:
            x: Array of shape ``(..., N_FEATURES)``.

        Returns:
            Normalised array of the same shape.
        """
        result: _F32Array = ((x - self.mean) / self.std).astype(np.float32)
        return result

    def to_dict(self) -> dict[str, list[float]]:
        """Serialise to a plain dict suitable for JSON persistence.

        Returns:
            Dict with keys ``"mean"`` and ``"std"``.
        """
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, list[float]]) -> NormStats:
        """Reconstruct from a previously serialised dict.

        Args:
            d: Dict produced by :meth:`to_dict`.

        Returns:
            Reconstructed :class:`NormStats` instance.
        """
        return cls(
            mean=np.array(d["mean"], dtype=np.float32),
            std=np.array(d["std"], dtype=np.float32),
        )

    @classmethod
    def from_json(cls, path: str) -> NormStats:
        """Load from a JSON file written by :meth:`to_dict`.

        Args:
            path: Filesystem path to the JSON file.

        Returns:
            Reconstructed :class:`NormStats` instance.
        """
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def to_json(self, path: str) -> None:
        """Persist to a JSON file.

        Args:
            path: Destination filesystem path.
        """
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)


def _build_windows(
    group: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_len: int,
) -> tuple[list[_F32Array], list[float]]:
    """Slide a window over a single stint and return (window, target) pairs.

    Only stints with at least ``seq_len + 1`` laps produce any windows, which
    guarantees every window has a full history and a distinct future target.

    Args:
        group: DataFrame rows for one (year, round, driver, stint), sorted by lap.
        feature_cols: Ordered list of feature column names.
        target_col: Name of the regression target column.
        seq_len: Number of historical laps per window.

    Returns:
        Tuple of (list of arrays shape ``(seq_len, n_features)``,
                  list of scalar target values).
    """
    vals = group[feature_cols].to_numpy(dtype=np.float32)
    targets = group[target_col].to_numpy(dtype=np.float32)
    n = len(vals)

    windows: list[_F32Array] = []
    window_targets: list[float] = []

    if n < seq_len + 1:
        return windows, window_targets

    for t in range(seq_len, n):
        windows.append(vals[t - seq_len : t])
        window_targets.append(float(targets[t]))

    return windows, window_targets


class StintSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sliding-window stint dataset for sequence tire degradation models.

    Filters the gold DataFrame to clean race laps, groups by stint, and
    builds overlapping ``(window, target)`` pairs.  Normalisation statistics
    are computed on the training split and must be passed unchanged to the
    validation split so the two share a consistent feature scale.

    Args:
        df: Gold-layer DataFrame.  Must contain all columns in
            :data:`FEATURE_COLS` plus ``TARGET_COL`` and the filtering
            columns listed below.
        norm_stats: Pre-computed normalisation statistics.  When ``None``,
            statistics are fitted on the data visible after the train/val
            split — always ``None`` for the training set and always provided
            for the validation set.
        val_rounds: Round numbers reserved for validation.  ``None`` means no
            split is performed and all laps are kept.
        is_val: When ``True`` keep only laps whose ``round_number`` is in
            ``val_rounds``; when ``False`` exclude those rounds.
        seq_len: History window length in laps.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        norm_stats: NormStats | None = None,
        val_rounds: list[int] | None = None,
        is_val: bool = False,
        seq_len: int = SEQ_LEN,
    ) -> None:
        self._seq_len = seq_len
        clean = self._filter(df)

        if val_rounds is not None:
            mask = clean["round_number"].isin(val_rounds)
            clean = clean[mask] if is_val else clean[~mask]

        if norm_stats is None:
            arr = clean[FEATURE_COLS].to_numpy(dtype=np.float32)
            norm_stats = NormStats(
                mean=arr.mean(axis=0),
                std=arr.std(axis=0),
            )
        self.norm_stats: NormStats = norm_stats

        self._xs: list[torch.Tensor] = []
        self._ys: list[torch.Tensor] = []

        group_keys = ["year", "round_number", "driver_number", "stint_number"]
        for _, group in clean.sort_values([*group_keys, "lap_number"]).groupby(
            group_keys, sort=False
        ):
            windows, targets = _build_windows(group, FEATURE_COLS, TARGET_COL, seq_len)
            for w, t in zip(windows, targets, strict=True):
                normed = self.norm_stats.normalize(w)
                self._xs.append(torch.from_numpy(normed))
                self._ys.append(torch.tensor(t, dtype=torch.float32))

    @staticmethod
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        """Apply gold-layer quality filters identical to the LightGBM pipeline.

        Args:
            df: Raw gold DataFrame.

        Returns:
            Filtered copy with NaN fills applied.
        """
        mask = (
            (df["session"] == "R")
            & df["is_accurate"].astype(bool)
            & (~df["pit_in_this_lap"].astype(bool))
            & (~df["pit_out_this_lap"].astype(bool))
            & df["lap_delta_to_field_median_s"].notna()
            & df["tyre_life_laps"].notna()
            & (df["tyre_life_laps"] >= 1)
        )
        clean = df[mask].copy()
        clean["is_fresh_tyre"] = clean["is_fresh_tyre"].fillna(0).astype(np.float32)
        clean["tyre_deg_rate_s_per_lap"] = (
            clean["tyre_deg_rate_s_per_lap"].fillna(0.0).astype(np.float32)
        )
        # Fill NaN with neutral values before normalisation.
        # sc_laps_since_last: 50 ≈ "no SC in living memory"
        clean["sc_laps_since_last"] = (
            clean["sc_laps_since_last"].fillna(50.0).astype(np.float32)
        )
        # position: 10 ≈ midfield (avoids biasing toward front/back)
        clean["position"] = clean["position"].fillna(10.0).astype(np.float32)
        return clean

    def __len__(self) -> int:
        """Return the total number of (window, target) pairs."""
        return len(self._xs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the idx-th (window, target) pair.

        Args:
            idx: Sample index.

        Returns:
            Tuple of ``(x, y)`` where ``x`` has shape ``(seq_len, N_FEATURES)``
            and ``y`` is a scalar tensor.
        """
        return self._xs[idx], self._ys[idx]


def compute_r2(preds: _F32Array, targets: _F32Array) -> float:
    """Coefficient of determination (R²) between predictions and targets.

    Args:
        preds: Predicted values, shape ``(n,)``.
        targets: Ground-truth values, shape ``(n,)``.

    Returns:
        R² score (1.0 is perfect; can be negative for very bad models).
    """
    ss_res = float(np.sum((targets - preds) ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def to_arch_config(
    seq_len: int,
    n_features: int,
    d_model: int,
    **kwargs: Any,
) -> dict[str, Any]:
    """Assemble a model architecture config dict for persistence.

    Args:
        seq_len: Input sequence length.
        n_features: Number of features per time step.
        d_model: Transformer / TCN model dimension.
        **kwargs: Any additional architecture hyper-parameters.

    Returns:
        Dict suitable for ``json.dump``.
    """
    return {"seq_len": seq_len, "n_features": n_features, "d_model": d_model, **kwargs}
