"""DQN+PER strategy policy — inference.

Loads the trained DQNPER policy from an MLflow artifact and provides the
same action-recommendation interface as ``predict.py`` (PPO), so the serving
layer can swap models without changing calling code.

Q-values are exposed alongside action labels — unlike a stochastic PPO
policy, DQN makes its value estimates directly interpretable.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import mlflow
import mlflow.tracking
import numpy as np
import numpy.typing as npt
import torch as th
from dotenv import load_dotenv

from ml.models.strategy_policy.per_buffer import PrioritizedReplayBuffer
from ml.models.strategy_policy.train_dqn import DQNPER

load_dotenv()

ACTION_NAMES: dict[int, str] = {
    0: "Stay out",
    1: "Pit — SOFT",
    2: "Pit — MEDIUM",
    3: "Pit — HARD",
}

COMPOUND_NAMES: dict[int, str] = {0: "SOFT", 1: "MEDIUM", 2: "HARD"}

_MLFLOW_EXPERIMENT = "strategy-policy"


def load_policy(run_id: str | None = None) -> DQNPER:
    """Load the trained DQN+PER policy from a local path or MLflow artifact.

    When ``PITWALL_DQN_POLICY_PATH`` env var is set (e.g. in the Docker image),
    the policy is loaded directly from that file.  Otherwise the most recent
    run in the ``strategy-policy`` MLflow experiment is used.

    Args:
        run_id: MLflow run ID to load from.  If ``None``, uses the most recent
            run in the experiment.  Ignored when ``PITWALL_DQN_POLICY_PATH``
            is set.

    Returns:
        Loaded DQNPER model ready for inference (CPU device).

    Raises:
        RuntimeError: If no experiment or runs are found.
    """
    local_path = os.getenv("PITWALL_DQN_POLICY_PATH")
    if local_path:
        return DQNPER.load(
            local_path,
            device="cpu",
            custom_objects={"replay_buffer_class": PrioritizedReplayBuffer},
        )

    tracking_uri = os.getenv("PITWALL_MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    if run_id is None:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(_MLFLOW_EXPERIMENT)
        if experiment is None:
            raise RuntimeError(
                f"MLflow experiment '{_MLFLOW_EXPERIMENT}' not found. "
                "Run `make train-policy-dqn` first."
            )
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.registered_model_name = 'pitwall-strategy-policy-dqn'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise RuntimeError(
                "No DQN training runs found. Run `make train-policy-dqn` first."
            )
        run_id = runs[0].info.run_id

    with tempfile.TemporaryDirectory() as tmpdir:
        local_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model",
            dst_path=tmpdir,
        )
        policy_path = os.path.join(local_dir, "policy")
        return DQNPER.load(
            policy_path,
            device="cpu",
            custom_objects={"replay_buffer_class": PrioritizedReplayBuffer},
        )


def recommend_action(
    obs: npt.NDArray[Any],
    model: DQNPER,
) -> tuple[int, str]:
    """Get the greedy pit-stop recommendation for the current race state.

    DQN is always deterministic at inference (argmax over Q-values).

    Args:
        obs: 21-dimensional observation vector from :class:`F1RaceEnv`.
        model: Loaded DQNPER model from :func:`load_policy`.

    Returns:
        Tuple of ``(action_int, action_label)`` where ``action_int`` ∈ {0,1,2,3}.

    Raises:
        ValueError: If the observation dimension does not match training.
    """
    obs_arr = np.array(obs, dtype=np.float32).reshape(1, -1)
    assert model.observation_space.shape is not None
    expected = model.observation_space.shape[0]
    if obs_arr.shape[1] != expected:
        raise ValueError(
            f"Observation has {obs_arr.shape[1]} features but this policy was "
            f"trained on {expected}. Re-train: "
            f"uv run python -m ml.models.strategy_policy.train_dqn main"
        )
    action, _ = model.predict(obs_arr, deterministic=True)
    action_int = int(action[0] if hasattr(action, "__len__") else action)
    return action_int, ACTION_NAMES[action_int]


def q_values(obs: npt.NDArray[Any], model: DQNPER) -> dict[str, float]:
    """Return raw Q-values for all four actions.

    Q-values represent the expected cumulative reward from each action in the
    current state — directly inspectable unlike a stochastic policy's
    probabilities.  Useful for confidence visualisation in the Streamlit UI.

    Args:
        obs: 21-dimensional observation vector.
        model: Loaded DQNPER model.

    Returns:
        Dict mapping action label → Q-value (not normalised).
    """
    obs_arr = np.array(obs, dtype=np.float32).reshape(1, -1)
    obs_tensor = model.policy.obs_to_tensor(obs_arr)[0]
    with th.no_grad():
        q = model.q_net(obs_tensor).squeeze(0)
    return {ACTION_NAMES[i]: float(q[i].item()) for i in range(len(ACTION_NAMES))}


def action_probabilities(obs: npt.NDArray[Any], model: DQNPER) -> dict[str, float]:
    """Return softmax-normalised Q-values as pseudo-probabilities.

    Softmax over Q-values is not a calibrated probability but gives an
    intuitive confidence score consistent with the ``predict.py`` PPO API.

    Args:
        obs: 21-dimensional observation vector.
        model: Loaded DQNPER model.

    Returns:
        Dict mapping action label → probability (sums to ≈1.0).
    """
    obs_arr = np.array(obs, dtype=np.float32).reshape(1, -1)
    obs_tensor = model.policy.obs_to_tensor(obs_arr)[0]
    with th.no_grad():
        q = model.q_net(obs_tensor).squeeze(0)
        probs = th.softmax(q, dim=0).cpu().numpy()
    return {ACTION_NAMES[i]: float(probs[i]) for i in range(len(ACTION_NAMES))}
