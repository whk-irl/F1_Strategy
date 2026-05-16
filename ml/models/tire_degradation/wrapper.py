"""SequenceTireWrapper — batch inference adapter for sequence tire models.

Exposes the same ``.predict(df)`` interface as an MLflow pyfunc model so
``F1RaceEnv`` and the Streamlit app can use TCN+GRU / PatchTST without
knowing the internal model API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ml.models.tire_degradation.sequence_dataset import (
    FEATURE_COLS,
    N_FEATURES,
    SEQ_LEN,
    NormStats,
)

import numpy.typing as npt

_F32Array = npt.NDArray[np.float32]


class SequenceTireWrapper:
    """Exposes a sequence tire model via the MLflow pyfunc ``.predict(df)`` interface.

    ``F1RaceEnv`` calls ``tire_model.predict(df)`` to batch-precompute a tire
    degradation lookup table at env init.  This wrapper vectorises the
    steady-state approximation so the env needs no changes.

    Args:
        model: Trained sequence model in eval mode.
        norm_stats: Normalisation statistics matching the model.
    """

    def __init__(self, model: nn.Module, norm_stats: NormStats) -> None:
        self._model = model
        self._ns = norm_stats

    def predict(self, df: pd.DataFrame) -> _F32Array:
        """Batch-predict lap-time deltas for a feature DataFrame.

        Each row is treated as a steady-state tyre state: the single step is
        repeated ``SEQ_LEN`` times to form the history window.  One model
        forward pass handles all rows simultaneously.

        Args:
            df: DataFrame with columns matching :data:`FEATURE_COLS` (unknown
                columns are filled with zeros; missing new columns use neutral
                defaults).

        Returns:
            Float32 array of shape ``(len(df),)`` with predicted deltas.
        """
        n = len(df)
        step = np.zeros((n, N_FEATURES), dtype=np.float32)
        neutral = {"sc_laps_since_last": 50.0, "position": 10.0}
        for col_idx, col in enumerate(FEATURE_COLS):
            if col in df.columns:
                step[:, col_idx] = df[col].to_numpy(dtype=np.float32)
            elif col in neutral:
                step[:, col_idx] = neutral[col]
        # Tile each row into (SEQ_LEN, N_FEATURES) — single forward pass.
        windows = np.tile(step[:, np.newaxis, :], (1, SEQ_LEN, 1))
        normed = (windows - self._ns.mean) / (self._ns.std + 1e-8)
        x = torch.from_numpy(normed.astype(np.float32))
        with torch.no_grad():
            preds = self._model(x).numpy()
        return preds.astype(np.float32)
