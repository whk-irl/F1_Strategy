"""Load a versioned history-aware policy bundle for inference."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from services.ingestion.acquire import file_sha256
from stable_baselines3 import PPO


class StagedPolicyPredictor:
    """Maintain one driver's lap history; call reset when switching race or driver."""

    def __init__(self, directory: str | Path) -> None:
        root = Path(directory)
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if (
            self.manifest.get("format_version") != 1
            or self.manifest.get("features_per_frame") != 24
        ):
            raise ValueError("Unsupported policy bundle observation schema")
        relative = self.manifest["policy"]
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Policy artifact must stay inside its bundle")
        digests = {key.replace("\\", "/"): value for key, value in self.manifest["sha256"].items()}
        if file_sha256(path) != digests[relative]:
            raise ValueError("Policy artifact checksum mismatch")
        self.model = PPO.load(str(path), device="cpu")
        self.frames = int(self.manifest["frames"])
        if self.model.observation_space.shape != (24 * self.frames,):
            raise ValueError("Checkpoint and manifest disagree on observation dimensions")
        self.history: deque[np.ndarray] = deque(maxlen=self.frames)
        self.reset()

    def reset(self) -> None:
        """Start a new episode with zero padding, matching training exactly."""
        self.history.clear()
        self.history.extend(np.zeros(24, dtype=np.float32) for _ in range(self.frames - 1))

    def recommend(self, observation: np.ndarray, used_compounds: set[int]) -> dict[str, Any]:
        """Recommend a dry action using a simulator-layout 21-value observation."""
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (21,) or not np.isfinite(obs).all():
            raise ValueError("Expected 21 finite observation values")
        if obs[18] > 0 or obs[19] > 0.3:
            raise ValueError("This candidate supports dry-race actions only")
        if not used_compounds or not used_compounds <= {0, 1, 2}:
            raise ValueError("Provide the dry compounds used by this driver so far")
        frame = np.concatenate(
            [obs, np.array([i in used_compounds for i in range(3)], dtype=np.float32)]
        )
        self.history.append(frame)
        stacked = np.concatenate(list(self.history)).astype(np.float32)
        action = int(self.model.predict(stacked, deterministic=True)[0])
        return {
            "action": action,
            "recommendation": ["stay", "pit_soft", "pit_medium", "pit_hard"][action],
            "promoted": self.manifest["promoted"],
        }
