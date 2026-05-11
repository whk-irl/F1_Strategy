"""Strategy policy model — inference.

Loads the trained PPO policy from an MLflow artifact and provides a
simple interface for pit-stop action recommendations given live race state.
"""

from __future__ import annotations

import os
import tempfile

import mlflow
import mlflow.tracking
import numpy as np
from dotenv import load_dotenv
from stable_baselines3 import PPO

load_dotenv()  # picks up PITWALL_MLFLOW_TRACKING_URI from .env

ACTION_NAMES: dict[int, str] = {
    0: "Stay out",
    1: "Pit — SOFT",
    2: "Pit — MEDIUM",
    3: "Pit — HARD",
}

COMPOUND_NAMES: dict[int, str] = {0: "SOFT", 1: "MEDIUM", 2: "HARD"}


def load_policy(run_id: str | None = None) -> PPO:
    """Load the trained PPO policy from a local path or MLflow run artifact.

    When ``PITWALL_POLICY_PATH`` env var is set (e.g. in the Docker image),
    the policy is loaded directly from that file, bypassing MLflow entirely.
    Otherwise the latest run in the ``strategy-policy`` experiment is used.

    Args:
        run_id: MLflow run ID to load from.  If None, uses the most recent
            run in the ``strategy-policy`` experiment.  Ignored when
            ``PITWALL_POLICY_PATH`` is set.

    Returns:
        Loaded Stable-Baselines3 PPO model ready for inference.

    Raises:
        RuntimeError: If no experiment or runs are found.
    """
    local_path = os.getenv("PITWALL_POLICY_PATH")
    if local_path:
        return PPO.load(local_path, device="cpu")

    tracking_uri = os.getenv("PITWALL_MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    if run_id is None:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("strategy-policy")
        if experiment is None:
            raise RuntimeError(
                "MLflow experiment 'strategy-policy' not found. Run `make train-policy` first."
            )
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise RuntimeError("No completed training runs found. Run `make train-policy` first.")
        run_id = runs[0].info.run_id

    with tempfile.TemporaryDirectory() as tmpdir:
        local_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model",
            dst_path=tmpdir,
        )
        policy_path = os.path.join(local_dir, "policy")
        return PPO.load(policy_path, device="cpu")


def recommend_action(
    obs: np.ndarray,
    model: PPO,
    *,
    deterministic: bool = True,
) -> tuple[int, str]:
    """Get the policy's pit-stop recommendation for the current race state.

    Args:
        obs: 21-dimensional observation vector from :class:`F1RaceEnv`.
            See ``env.py`` for the full feature description.
        model: Loaded PPO model from :func:`load_policy`.
        deterministic: Use greedy action (True for deployment), stochastic
            for diversity during evaluation.

    Returns:
        Tuple of ``(action_int, action_label)`` where ``action_int`` is in
        ``{0, 1, 2, 3}`` and ``action_label`` is a human-readable string.

    Raises:
        ValueError: If the observation dimension does not match what the model
            was trained on — typically means you loaded a stale checkpoint that
            predates the current env observation space.
    """
    obs_arr = np.array(obs, dtype=np.float32).reshape(1, -1)
    assert model.observation_space.shape is not None
    expected = model.observation_space.shape[0]
    if obs_arr.shape[1] != expected:
        raise ValueError(
            f"Observation has {obs_arr.shape[1]} features but this policy was "
            f"trained on {expected}. Re-train the policy with the current env: "
            f"uv run python -m ml.models.strategy_policy.train main --timesteps 3000000"
        )
    action, _ = model.predict(obs_arr, deterministic=deterministic)
    action_int = int(action[0] if hasattr(action, "__len__") else action)
    return action_int, ACTION_NAMES[action_int]


def action_probabilities(
    obs: np.ndarray,
    model: PPO,
) -> dict[str, float]:
    """Return softmax action probabilities for all four pit decisions.

    Useful for explaining why the policy recommends a particular action.

    Args:
        obs: 21-dimensional observation vector.
        model: Loaded PPO model.

    Returns:
        Dict mapping action label → probability (sums to 1.0).
    """
    import torch

    obs_arr = np.array(obs, dtype=np.float32).reshape(1, -1)
    assert model.observation_space.shape is not None
    expected = model.observation_space.shape[0]
    if obs_arr.shape[1] != expected:
        raise ValueError(
            f"Observation has {obs_arr.shape[1]} features but this policy was "
            f"trained on {expected}. Re-train: "
            f"uv run python -m ml.models.strategy_policy.train main --timesteps 3000000"
        )
    obs_tensor = model.policy.obs_to_tensor(obs_arr)[0]
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.squeeze().cpu().numpy()  # type: ignore[union-attr]

    return {ACTION_NAMES[i]: float(p) for i, p in enumerate(probs)}
