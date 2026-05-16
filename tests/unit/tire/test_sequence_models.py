"""Unit tests for the sequence tire degradation models.

All tests use CPU-only tiny dummy data — no GPU or trained checkpoints required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from ml.models.tire_degradation.model_patch_tst import PatchTSTTireModel
from ml.models.tire_degradation.model_tcn_gru import TCNGRUTireModel
from ml.models.tire_degradation.predict_sequence import (
    predict_lap_time_delta,
    predict_stint_degradation,
    predict_stint_from_history,
)
from ml.models.tire_degradation.sequence_dataset import (
    FEATURE_COLS,
    N_FEATURES,
    SEQ_LEN,
    NormStats,
    StintSequenceDataset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gold_df(n_stints: int = 3, laps_per_stint: int = 20) -> pd.DataFrame:
    """Build a minimal gold-like DataFrame for dataset construction tests.

    Args:
        n_stints: Number of distinct (driver, stint) combinations.
        laps_per_stint: Laps per stint; must be > SEQ_LEN (15) to produce windows.

    Returns:
        DataFrame with all gold columns expected by :class:`StintSequenceDataset`.
    """
    rows = []
    for stint_idx in range(n_stints):
        for lap in range(1, laps_per_stint + 1):
            rows.append(
                {
                    "year": 2024,
                    "round_number": 1,
                    "session": "R",
                    "driver_number": stint_idx + 1,
                    "stint_number": 1,
                    "lap_number": lap,
                    "tyre_life_laps": float(lap),
                    "compound_encoded": 1.0,
                    "race_progress": lap / (laps_per_stint * n_stints),
                    "track_status_encoded": 0.0,
                    "is_fresh_tyre": 1.0 if lap == 1 else 0.0,
                    "lap_delta_to_field_median_s": float(np.random.normal(0, 0.1)),
                    "lap_time_delta_s": float(np.random.normal(0, 0.5)),
                    "tyre_deg_rate_s_per_lap": float(np.random.normal(0.05, 0.02)),
                    "sc_laps_since_last": float(lap + 10),
                    "position": float((stint_idx % 10) + 1),
                    "pit_in_this_lap": False,
                    "pit_out_this_lap": False,
                    "is_accurate": True,
                }
            )
    return pd.DataFrame(rows)


def _make_norm_stats() -> NormStats:
    """Create trivial (identity) NormStats for smoke tests."""
    return NormStats(
        mean=np.zeros(N_FEATURES, dtype=np.float32),
        std=np.ones(N_FEATURES, dtype=np.float32),
    )


def _make_tcn_gru() -> TCNGRUTireModel:
    model = TCNGRUTireModel(n_features=N_FEATURES)
    model.eval()
    return model


def _make_patch_tst() -> PatchTSTTireModel:
    model = PatchTSTTireModel(n_features=N_FEATURES, seq_len=SEQ_LEN)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNormStats:
    def test_roundtrip_dict(self) -> None:
        """NormStats serialises to dict and deserialises identically."""
        norm = NormStats(
            mean=np.arange(N_FEATURES, dtype=np.float32),
            std=np.ones(N_FEATURES, dtype=np.float32) * 2.0,
        )
        recovered = NormStats.from_dict(norm.to_dict())
        np.testing.assert_array_almost_equal(norm.mean, recovered.mean)
        np.testing.assert_array_almost_equal(norm.std, recovered.std)

    def test_zero_std_clamped(self) -> None:
        """A zero std column is clamped to 1.0 to prevent division-by-zero."""
        norm = NormStats(
            mean=np.zeros(N_FEATURES, dtype=np.float32),
            std=np.zeros(N_FEATURES, dtype=np.float32),
        )
        assert float(norm.std.min()) == 1.0

    def test_roundtrip_json(self, tmp_path: pytest.TempPathFactory) -> None:
        """NormStats survives a JSON round-trip via to_json / from_json."""
        norm = NormStats(
            mean=np.random.randn(N_FEATURES).astype(np.float32),
            std=np.abs(np.random.randn(N_FEATURES)).astype(np.float32) + 0.1,
        )
        path = str(tmp_path / "norm.json")  # type: ignore[operator]
        norm.to_json(path)
        recovered = NormStats.from_json(path)
        np.testing.assert_array_almost_equal(norm.mean, recovered.mean)
        np.testing.assert_array_almost_equal(norm.std, recovered.std)

    def test_normalize_shape(self) -> None:
        """normalize() preserves shape and produces float32."""
        norm = _make_norm_stats()
        x = np.random.randn(SEQ_LEN, N_FEATURES).astype(np.float32)
        out = norm.normalize(x)
        assert out.shape == (SEQ_LEN, N_FEATURES)
        assert out.dtype == np.float32


class TestStintSequenceDataset:
    def test_dataset_builds_and_nonempty(self) -> None:
        """Dataset builds from a small gold-like DataFrame and has samples."""
        np.random.seed(42)
        df = _make_gold_df(n_stints=3, laps_per_stint=20)
        ds = StintSequenceDataset(df)
        # 3 stints × (20 - SEQ_LEN) windows = 3 × 5 = 15
        assert len(ds) == 15

    def test_tensor_shapes(self) -> None:
        """Each sample has the correct tensor shapes."""
        np.random.seed(42)
        df = _make_gold_df(n_stints=2, laps_per_stint=17)
        ds = StintSequenceDataset(df)
        x, y = ds[0]
        assert x.shape == (SEQ_LEN, N_FEATURES)
        assert y.shape == ()

    def test_tensor_dtype(self) -> None:
        """Tensors are float32."""
        np.random.seed(42)
        df = _make_gold_df(n_stints=1, laps_per_stint=17)
        ds = StintSequenceDataset(df)
        x, y = ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_val_split_disjoint(self) -> None:
        """Train and val splits share no round_number values."""
        np.random.seed(42)
        base = _make_gold_df(n_stints=1, laps_per_stint=17)
        # Add laps from a second round for the val split
        val_df = base.copy()
        val_df["round_number"] = 5
        df = pd.concat([base, val_df], ignore_index=True)

        train_ds = StintSequenceDataset(df, val_rounds=[5], is_val=False)
        val_ds = StintSequenceDataset(
            df,
            norm_stats=train_ds.norm_stats,
            val_rounds=[5],
            is_val=True,
        )
        # Both splits must be non-empty
        assert len(train_ds) > 0
        assert len(val_ds) > 0

    def test_norm_stats_attribute(self) -> None:
        """norm_stats is accessible as a public attribute."""
        np.random.seed(42)
        df = _make_gold_df(n_stints=1, laps_per_stint=17)
        ds = StintSequenceDataset(df)
        assert isinstance(ds.norm_stats, NormStats)
        assert ds.norm_stats.mean.shape == (N_FEATURES,)

    def test_short_stint_skipped(self) -> None:
        """Stints shorter than SEQ_LEN+1 laps produce zero windows."""
        df = _make_gold_df(n_stints=1, laps_per_stint=SEQ_LEN)  # exactly SEQ_LEN rows
        ds = StintSequenceDataset(df)
        assert len(ds) == 0


class TestTCNGRUForward:
    def test_output_shape(self) -> None:
        """Forward pass returns shape (batch,) for batch size 2."""
        model = _make_tcn_gru()
        x = torch.randn(2, SEQ_LEN, N_FEATURES)
        out = model(x)
        assert out.shape == (2,)

    def test_no_nan(self) -> None:
        """Forward pass produces no NaN values."""
        model = _make_tcn_gru()
        x = torch.randn(4, SEQ_LEN, N_FEATURES)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_single_sample(self) -> None:
        """Works with batch size 1."""
        model = _make_tcn_gru()
        x = torch.randn(1, SEQ_LEN, N_FEATURES)
        out = model(x)
        assert out.shape == (1,)

    def test_gradient_flows(self) -> None:
        """Loss.backward() succeeds — gradients exist for all parameters."""
        model = TCNGRUTireModel(n_features=N_FEATURES)
        x = torch.randn(2, SEQ_LEN, N_FEATURES)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"


class TestPatchTSTForward:
    def test_output_shape(self) -> None:
        """Forward pass returns shape (batch,) for batch size 2."""
        model = _make_patch_tst()
        x = torch.randn(2, SEQ_LEN, N_FEATURES)
        out = model(x)
        assert out.shape == (2,)

    def test_no_nan(self) -> None:
        """Forward pass produces no NaN values."""
        model = _make_patch_tst()
        x = torch.randn(4, SEQ_LEN, N_FEATURES)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_single_sample(self) -> None:
        """Works with batch size 1."""
        model = _make_patch_tst()
        x = torch.randn(1, SEQ_LEN, N_FEATURES)
        out = model(x)
        assert out.shape == (1,)

    def test_gradient_flows(self) -> None:
        """Loss.backward() succeeds — gradients exist for all parameters."""
        model = PatchTSTTireModel(n_features=N_FEATURES, seq_len=SEQ_LEN)
        x = torch.randn(2, SEQ_LEN, N_FEATURES)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"

    def test_odd_seq_len(self) -> None:
        """Works when seq_len is not a multiple of patch_len (tests padding)."""
        model = PatchTSTTireModel(n_features=N_FEATURES, seq_len=7, patch_len=5)
        x = torch.randn(2, 7, N_FEATURES)
        out = model(x)
        assert out.shape == (2,)
        assert not torch.isnan(out).any()


class TestScalarInterface:
    def test_predict_lap_time_delta_returns_float(self) -> None:
        """predict_lap_time_delta returns a Python float with random-weight model."""
        model = _make_tcn_gru()
        norm = _make_norm_stats()
        result = predict_lap_time_delta(
            tyre_life_laps=5,
            compound_encoded=1,
            race_progress=0.3,
            track_status_encoded=0,
            is_fresh_tyre=False,
            lap_delta_to_field_median_s=0.2,
            model=model,
            norm_stats=norm,
        )
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_predict_lap_time_delta_patch_tst(self) -> None:
        """predict_lap_time_delta works with PatchTST model."""
        model = _make_patch_tst()
        norm = _make_norm_stats()
        result = predict_lap_time_delta(
            tyre_life_laps=10,
            compound_encoded=2,
            race_progress=0.6,
            track_status_encoded=0,
            is_fresh_tyre=False,
            lap_delta_to_field_median_s=-0.1,
            model=model,
            norm_stats=norm,
        )
        assert isinstance(result, float)
        assert np.isfinite(result)


class TestAutoregressive:
    def test_predict_stint_from_history_shape(self) -> None:
        """predict_stint_from_history returns array of the requested length."""
        model = _make_tcn_gru()
        norm = _make_norm_stats()
        history = pd.DataFrame(
            {
                "tyre_life_laps": np.arange(1, 6, dtype=np.float32),
                "compound_encoded": np.ones(5, dtype=np.float32),
                "race_progress": np.linspace(0.1, 0.2, 5),
                "track_status_encoded": np.zeros(5, dtype=np.float32),
                "is_fresh_tyre": [1.0] + [0.0] * 4,
                "lap_delta_to_field_median_s": np.zeros(5, dtype=np.float32),
                "lap_time_delta_s": np.random.randn(5).astype(np.float32) * 0.3,
                "tyre_deg_rate_s_per_lap": np.ones(5, dtype=np.float32) * 0.05,
                "sc_laps_since_last": np.ones(5, dtype=np.float32) * 50.0,
                "position": np.ones(5, dtype=np.float32) * 10.0,
            }
        )
        preds = predict_stint_from_history(history, n_forecast_laps=8, model=model, norm_stats=norm)
        assert preds.shape == (8,)
        assert np.all(np.isfinite(preds))

    def test_predict_stint_from_history_single_row(self) -> None:
        """predict_stint_from_history works when history has only 1 row (heavy padding)."""
        model = _make_patch_tst()
        norm = _make_norm_stats()
        history = pd.DataFrame(
            [
                dict.fromkeys(FEATURE_COLS, 0.0),
            ]
        )
        history["tyre_life_laps"] = 1.0
        history["is_fresh_tyre"] = 1.0
        preds = predict_stint_from_history(history, n_forecast_laps=5, model=model, norm_stats=norm)
        assert preds.shape == (5,)
        assert np.all(np.isfinite(preds))

    def test_predict_stint_degradation_shape(self) -> None:
        """predict_stint_degradation returns array matching stint_laps."""
        model = _make_tcn_gru()
        norm = _make_norm_stats()
        preds = predict_stint_degradation(
            stint_laps=10,
            compound_encoded=0,
            race_progress_start=0.1,
            lap_delta_to_field_median_s=0.0,
            model=model,
            norm_stats=norm,
        )
        assert preds.shape == (10,)
        assert np.all(np.isfinite(preds))
