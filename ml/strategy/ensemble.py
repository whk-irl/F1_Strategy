"""Serve PPO pit policies as an ensemble (mean action probabilities across seeds).

Seeds trained identically differ by up to 2.6 s/race on held-out races; averaging
their action distributions removes that variance without selecting a seed on
test data. On 2024-25 the 5-seed HPC ensemble scored 1.68 s/race better than the
DP planner (95% race-bootstrap CI [-3.49, +0.17]); see
docs/guides/strategy-planner-20261105.md.

    policy = EnsemblePolicy.load(sorted(Path("runs/strategy-ppo-186536").glob("seed-*/policy.zip")))
    action = policy.act(observation(lap, n, compound, age, two_used, neutral, estimates))
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO


class EnsemblePolicy:
    """Argmax of the mean action probabilities of several PPO policies."""

    def __init__(self, models: Sequence[PPO]) -> None:
        if not models:
            raise ValueError("An ensemble needs at least one policy")
        self.models = list(models)

    @classmethod
    def load(cls, paths: Sequence[Path | str], device: str = "cpu") -> EnsemblePolicy:
        return cls([PPO.load(str(p), device=device) for p in paths])

    def probabilities(self, obs: np.ndarray) -> np.ndarray:
        """Mean action probabilities (stay, SOFT, MEDIUM, HARD)."""
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs, dtype=np.float32)[None])
            probs = [m.policy.get_distribution(x).distribution.probs for m in self.models]
        return np.asarray(torch.stack(probs).mean(dim=0)[0])

    def act(self, obs: np.ndarray) -> int:
        return int(np.argmax(self.probabilities(obs)))
