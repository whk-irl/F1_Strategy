"""DQN + Prioritized Experience Replay strategy policy training.

Replaces the PPO policy (``train.py``) with a DQN variant that uses
Prioritized Experience Replay (Schaul et al. 2016) to focus learning on
high-TD-error transitions — primarily the SC/undercut windows that matter
most for race strategy.

PPO was collapsing to a degenerate "pit at lap 1" policy because the
long-range discount signal cannot outweigh the immediate cliff-penalty
relief from fresh tyres.  DQN's explicit Q-value table makes the
per-compound value inspectable and its off-policy replay lets it revisit
rare SC events more frequently via priority weighting.

See ``docs/decisions/0003-dqn-per-strategy-policy.md`` for the full ADR.

Prerequisites (run in order):
    make ingest-season SEASON=2024
    make train-tire
    make train-safety-car
    make validate-simulator   ← must pass ρ ≥ 0.7

Entry point:
    python -m ml.models.strategy_policy.train_dqn
    make train-policy-dqn
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Annotated, Any, ClassVar

import gymnasium as gym
import mlflow
import mlflow.pyfunc
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch as th
import torch.nn.functional as F  # noqa: N812
import typer
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from services.simulator.env import F1RaceEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import DummyVecEnv

from ml.models._loader import load_gold_seasons
from ml.models.strategy_policy.per_buffer import (
    PrioritizedReplayBuffer,
    PrioritizedReplayBufferSamples,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class StrategyDQNConfig(BaseSettings):
    """Hyper-parameters and data settings for DQN+PER strategy policy training."""

    model_config = SettingsConfigDict(env_prefix="PITWALL_DQN_", env_file=".env", extra="ignore")

    training_seasons: list[int] = Field(default=[2024])
    total_timesteps: int = Field(default=2_000_000)

    # DQN core
    learning_rate: float = Field(default=1e-4)
    buffer_size: int = Field(default=100_000)
    learning_starts: int = Field(default=10_000)
    batch_size: int = Field(default=64)
    gamma: float = Field(default=0.995)
    train_freq: int = Field(default=4)  # gradient step every N env steps
    gradient_steps: int = Field(default=1)
    target_update_interval: int = Field(default=10_000)
    tau: float = Field(default=1.0)  # hard target update (τ=1)
    max_grad_norm: float = Field(default=10.0)

    # Exploration schedule
    exploration_fraction: float = Field(default=0.3)
    exploration_initial_eps: float = Field(default=1.0)
    exploration_final_eps: float = Field(default=0.05)

    # PER
    per_alpha: float = Field(default=0.6)
    per_beta: float = Field(default=0.4)

    # Environment
    n_envs: int = Field(default=1)  # DQN is single-env by design
    min_drivers_per_race: int = Field(default=10)
    pit_loss_s: float = Field(default=25.0)
    noise_std: float = Field(default=0.2)

    mlflow_tracking_uri: str = Field(
        default="mlruns", validation_alias="PITWALL_MLFLOW_TRACKING_URI"
    )
    mlflow_experiment: str = Field(default="strategy-policy")
    registered_model_name: str = Field(default="pitwall-strategy-policy-dqn")

    tire_model_uri: str = Field(
        default="models:/pitwall-tire-degradation/latest",
        validation_alias="PITWALL_TIRE_MODEL_URI",
    )
    sc_model_uri: str = Field(
        default="models:/pitwall-safety-car/latest",
        validation_alias="PITWALL_SC_MODEL_URI",
    )


# ---------------------------------------------------------------------------
# Multi-race environment wrapper (same design as train.py)
# ---------------------------------------------------------------------------


class MultiRaceEnv(gym.Env[npt.NDArray[Any], npt.NDArray[Any]]):
    """F1RaceEnv wrapper that samples a random (race, driver) pair on each reset."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}  # type: ignore[misc]

    def __init__(
        self,
        race_pool: list[tuple[pd.DataFrame, Any]],
        tire_model: Any,
        sc_model: Any | None,
        pit_loss_s: float = 25.0,
        noise_std: float = 0.2,
    ) -> None:
        super().__init__()
        if not race_pool:
            raise ValueError("race_pool must not be empty")
        self._pool = race_pool
        first_df, first_drv = race_pool[0]
        self._inner = F1RaceEnv(first_df, first_drv, tire_model, sc_model, pit_loss_s, noise_std)
        self.action_space = self._inner.action_space
        self.observation_space = self._inner.observation_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[Any], dict[str, Any]]:
        super().reset(seed=seed)
        idx = int(self.np_random.integers(0, len(self._pool)))
        race_df, driver = self._pool[idx]
        return self._inner.reset(
            seed=seed,
            options={"race_df": race_df, "driver_number": driver},
        )

    def step(self, action: int) -> tuple[npt.NDArray[Any], float, bool, bool, dict[str, Any]]:  # type: ignore[override]
        return self._inner.step(action)


# ---------------------------------------------------------------------------
# Race pool (shared with train.py logic)
# ---------------------------------------------------------------------------


def build_race_pool(
    gold_df: pd.DataFrame,
    min_drivers: int = 10,
) -> list[tuple[pd.DataFrame, Any]]:
    """Build (race_df, driver_number) pairs for multi-race DQN training.

    Args:
        gold_df: Full gold DataFrame across one or more seasons.
        min_drivers: Minimum drivers with ≥10 complete laps for a race to qualify.

    Returns:
        List of (race_df, driver_number) tuples — one entry per controllable
        driver per qualifying race.
    """
    races_df = gold_df[gold_df["session"] == "R"]
    pool: list[tuple[pd.DataFrame, Any]] = []
    n_races = 0

    for (_year, _rnd), race_df in races_df.groupby(["year", "round_number"]):
        eligible = [
            drv
            for drv, grp in race_df.groupby("driver_number")
            if grp["lap_time_s"].notna().sum() >= 10
        ]
        if len(eligible) < min_drivers:
            continue
        shared_race_df = race_df.copy()
        for drv in eligible:
            pool.append((shared_race_df, drv))
        n_races += 1

    logger.info("Race pool: %d (race, driver) pairs across %d races.", len(pool), n_races)
    return pool


# ---------------------------------------------------------------------------
# DQNPER — DQN subclass with PER-weighted loss
# ---------------------------------------------------------------------------


class DQNPER(DQN):
    """DQN with Prioritized Experience Replay.

    Extends SB3's ``DQN`` by overriding ``train()`` to:

    1. Apply importance-sampling weights to the per-element Huber loss so
       high-priority transitions do not over-fit (bias correction).
    2. Compute TD errors after each gradient step and push them back to the
       ``PrioritizedReplayBuffer`` to keep priorities current.

    The constructor signature is identical to ``DQN``; simply pass
    ``replay_buffer_class=PrioritizedReplayBuffer`` and
    ``replay_buffer_kwargs`` with PER settings.
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """Override to weight the Huber loss by IS weights and update priorities."""
        assert isinstance(self.replay_buffer, PrioritizedReplayBuffer), (
            "DQNPER.train() requires a PrioritizedReplayBuffer. "
            "Pass replay_buffer_class=PrioritizedReplayBuffer to the constructor."
        )

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses: list[float] = []
        for _ in range(gradient_steps):
            replay_data: PrioritizedReplayBufferSamples = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            with th.no_grad():
                next_q = self.q_net_target(replay_data.next_observations)
                next_q, _ = next_q.max(dim=1)
                next_q = next_q.reshape(-1, 1)
                target_q = replay_data.rewards + (1.0 - replay_data.dones) * self.gamma * next_q

            current_q = self.q_net(replay_data.observations)
            current_q = th.gather(current_q, dim=1, index=replay_data.actions.long())

            # TD errors (no gradient) — fed back to the buffer
            td_errors = th.abs(current_q.detach() - target_q).squeeze(1)

            # IS-weighted Huber loss: weight large-TD transitions less to
            # correct for their over-representation in the sample.
            element_loss = F.smooth_l1_loss(current_q, target_q, reduction="none")
            loss = (element_loss * replay_data.weights.unsqueeze(1)).mean()
            losses.append(float(loss.item()))

            self.policy.optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            # Push updated priorities back to the buffer
            self.replay_buffer.update_priorities(
                replay_data.indices,
                td_errors.cpu().numpy(),
            )

        # Target network hard / soft update (SB3 checks gradient-step count)
        if self._n_updates % self.target_update_interval == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", float(np.mean(losses)))


# ---------------------------------------------------------------------------
# MLflow callback
# ---------------------------------------------------------------------------


class _MLflowCallback(BaseCallback):
    """Log mean episode reward and mean TD loss to MLflow every N timesteps.

    DQN's collect_rollouts fires _on_step once per env step (train_freq=4),
    so completed episodes are rare at any individual step.  We accumulate
    episode returns in a rolling buffer and flush the mean every log_freq
    steps so the metric is always populated.
    """

    def __init__(self, log_freq: int = 50_000) -> None:
        super().__init__()
        self._log_freq = log_freq
        self._last_logged = 0
        self._ep_returns: list[float] = []

    def _on_step(self) -> bool:
        # Accumulate any completed episodes from this step's infos
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_returns.append(float(info["episode"]["r"]))

        if self.num_timesteps - self._last_logged >= self._log_freq:
            if self._ep_returns:
                mlflow.log_metric(
                    "mean_episode_reward",
                    float(np.mean(self._ep_returns)),
                    step=self.num_timesteps,
                )
                self._ep_returns.clear()
            self._last_logged = self.num_timesteps
        return True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(config: StrategyDQNConfig | None = None) -> None:
    """Train the DQN+PER strategy policy and register it in MLflow.

    Args:
        config: Training configuration (defaults to env-vars / built-in defaults).
    """
    if config is None:
        config = StrategyDQNConfig()

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment)

    logger.info("Loading tire and SC models from MLflow registry...")
    tire_model = mlflow.pyfunc.load_model(config.tire_model_uri)
    sc_model = mlflow.pyfunc.load_model(config.sc_model_uri)

    logger.info("Loading gold data for seasons %s...", config.training_seasons)
    gold_df = load_gold_seasons(config.training_seasons)
    race_pool = build_race_pool(gold_df, config.min_drivers_per_race)

    if not race_pool:
        raise RuntimeError("Race pool is empty. Ingest data first or lower --min-drivers.")

    # DQN trains on a single env; multiple envs would cause replay-buffer index
    # ambiguity when updating priorities across workers.
    def _make_env() -> Monitor[Any, Any]:
        return Monitor(
            MultiRaceEnv(
                race_pool,
                tire_model,
                sc_model,
                config.pit_loss_s,
                config.noise_std,
            )
        )

    vec_env = DummyVecEnv([_make_env])

    per_beta_annealing_steps = config.total_timesteps // config.train_freq

    dqn_hparams: dict[str, Any] = {
        "learning_rate": config.learning_rate,
        "buffer_size": config.buffer_size,
        "learning_starts": config.learning_starts,
        "batch_size": config.batch_size,
        "gamma": config.gamma,
        "train_freq": config.train_freq,
        "gradient_steps": config.gradient_steps,
        "target_update_interval": config.target_update_interval,
        "tau": config.tau,
        "max_grad_norm": config.max_grad_norm,
        "exploration_fraction": config.exploration_fraction,
        "exploration_initial_eps": config.exploration_initial_eps,
        "exploration_final_eps": config.exploration_final_eps,
    }

    with mlflow.start_run():
        mlflow.log_params(
            {
                **dqn_hparams,
                "net_arch": "256x128",
                "algo": "DQN+PER",
                "per_alpha": config.per_alpha,
                "per_beta": config.per_beta,
                "per_beta_annealing_steps": per_beta_annealing_steps,
                "training_seasons": config.training_seasons,
                "total_timesteps": config.total_timesteps,
                "race_pool_size": len(race_pool),
                "pit_loss_s": config.pit_loss_s,
                "obs_dim": 21,
            }
        )

        model = DQNPER(
            policy="MlpPolicy",
            env=vec_env,
            policy_kwargs={"net_arch": [256, 128]},
            replay_buffer_class=PrioritizedReplayBuffer,
            replay_buffer_kwargs={
                "alpha": config.per_alpha,
                "beta": config.per_beta,
                "beta_annealing_steps": per_beta_annealing_steps,
            },
            device="auto",
            verbose=1,
            **dqn_hparams,
        )

        mlflow.log_param("device", str(model.device))
        logger.info(
            "Training DQN+PER on device: %s | %d timesteps",
            model.device,
            config.total_timesteps,
        )

        model.learn(
            total_timesteps=config.total_timesteps,
            callback=_MLflowCallback(log_freq=50_000),
            progress_bar=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = os.path.join(tmpdir, "policy")
            model.save(policy_path)
            mlflow.log_artifact(f"{policy_path}.zip", artifact_path="model")
            mlflow.set_tag("registered_model_name", config.registered_model_name)

        logger.info(
            "DQN+PER training complete — model logged to experiment '%s'.",
            config.mlflow_experiment,
        )

    vec_env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    seasons: Annotated[
        str,
        typer.Option("--seasons", help="Comma-separated seasons, e.g. 2022,2023,2024."),
    ] = "2024",
    timesteps: Annotated[
        int,
        typer.Option("--timesteps", help="Total DQN training timesteps."),
    ] = 2_000_000,
) -> None:
    """CLI entry point for DQN+PER strategy policy training."""
    season_list = [int(s.strip()) for s in seasons.split(",")]
    train(
        StrategyDQNConfig(
            training_seasons=season_list,
            total_timesteps=timesteps,
        )
    )


@app.command()
def finetune(
    checkpoint: Annotated[
        str,
        typer.Argument(help="Path to existing policy.zip (without the .zip extension)."),
    ],
    timesteps: Annotated[
        int,
        typer.Option("--timesteps", help="Additional timesteps to train."),
    ] = 200_000,
    seasons: Annotated[
        str,
        typer.Option("--seasons", help="Comma-separated seasons."),
    ] = "2024",
) -> None:
    """Fine-tune an existing DQN+PER checkpoint.

    Note: the replay buffer is not preserved across saves, so the agent will
    re-explore for ``learning_starts`` steps before gradient updates resume.
    """
    config = StrategyDQNConfig(
        training_seasons=[int(s.strip()) for s in seasons.split(",")],
        total_timesteps=timesteps,
    )

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment)

    tire_model = mlflow.pyfunc.load_model(config.tire_model_uri)
    sc_model = mlflow.pyfunc.load_model(config.sc_model_uri)

    gold_df = load_gold_seasons(config.training_seasons)
    race_pool = build_race_pool(gold_df, config.min_drivers_per_race)

    def _make_env() -> Monitor[Any, Any]:
        return Monitor(
            MultiRaceEnv(race_pool, tire_model, sc_model, config.pit_loss_s, config.noise_std)
        )

    vec_env = DummyVecEnv([_make_env])

    logger.info("Loading checkpoint: %s", checkpoint)
    model = DQNPER.load(
        checkpoint,
        env=vec_env,
        device="auto",
        custom_objects={
            "replay_buffer_class": PrioritizedReplayBuffer,
        },
    )

    with mlflow.start_run():
        mlflow.log_params({"finetune_checkpoint": checkpoint, "finetune_timesteps": timesteps})
        model.learn(
            total_timesteps=timesteps,
            callback=_MLflowCallback(log_freq=50_000),
            progress_bar=True,
            reset_num_timesteps=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = os.path.join(tmpdir, "policy")
            model.save(policy_path)
            mlflow.log_artifact(f"{policy_path}.zip", artifact_path="model")

    vec_env.close()
    logger.info("Fine-tuning complete.")


if __name__ == "__main__":
    app()
