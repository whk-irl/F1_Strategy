"""Shape, learning-objective, and checkpoint tests for temporal IQL."""

from __future__ import annotations

import numpy as np
import torch
from ml.pipelines.config import TrainingConfig
from ml.pipelines.iql import TemporalIQLPolicy, _expectile_loss


def test_expectile_weights_positive_residuals_more_heavily():
    residual = torch.tensor([-2.0, 2.0])
    loss = _expectile_loss(residual, 0.7)
    assert loss.item() == 2.0


def test_temporal_iql_predict_and_checkpoint_round_trip(tmp_path):
    config = TrainingConfig(
        history_frames=3,
        iql_hidden_dim=32,
        iql_heads=4,
        iql_layers=1,
        device="cpu",
    )
    width = config.history_frames * 24
    policy = TemporalIQLPolicy(
        config, np.zeros(width, dtype=np.float32), np.ones(width, dtype=np.float32)
    )
    observation = np.linspace(-1, 1, width, dtype=np.float32)
    expected = policy.predict(observation)[0]
    assert np.asarray(expected).shape == ()
    assert 0 <= int(expected) < 4
    checkpoint = tmp_path / "policy.pt"
    policy.save(checkpoint)
    restored = TemporalIQLPolicy.load(checkpoint)
    assert restored.predict(observation)[0] == expected


def test_temporal_iql_enforces_second_compound_near_finish():
    config = TrainingConfig(
        history_frames=2,
        iql_hidden_dim=32,
        iql_heads=4,
        iql_layers=1,
        device="cpu",
    )
    width = config.history_frames * 24
    policy = TemporalIQLPolicy(
        config, np.zeros(width, dtype=np.float32), np.ones(width, dtype=np.float32)
    )
    observation = np.zeros(width, dtype=np.float32)
    observation[-24 + 1] = 0.1
    observation[-3:] = [1, 0, 0]
    assert int(policy.predict(observation)[0]) in (2, 3)
