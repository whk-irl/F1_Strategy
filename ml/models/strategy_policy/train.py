"""PPO strategy policy training — Stable-Baselines3 + PyTorch + MLflow.

The policy learns when to pit and which compound to use by interacting with
the F1RaceEnv simulator across hundreds of races and driver combinations from
the gold data.  Training runs on GPU automatically (CUDA detected by PyTorch).

Prerequisites (run in order):
    make ingest-season SEASON=2024
    make train-tire
    make train-safety-car
    make validate-simulator   ← must pass ρ ≥ 0.7

Entry point:
    python -m ml.models.strategy_policy.train
    make train-policy
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
import pandas as pd
import typer
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from services.simulator.env import F1RaceEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ml.models._loader import load_gold_seasons

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class StrategyPolicyConfig(BaseSettings):
    """Hyper-parameters and data settings for PPO strategy policy training."""

    model_config = SettingsConfigDict(env_prefix="PITWALL_PPO_", env_file=".env", extra="ignore")

    training_seasons: list[int] = Field(default=[2024])
    total_timesteps: int = Field(default=1_000_000)
    n_envs: int = Field(default=4)
    n_steps: int = Field(default=2048)
    batch_size: int = Field(default=64)
    n_epochs: int = Field(default=10)
    learning_rate: float = Field(default=3e-4)
    gamma: float = Field(default=0.995)  # high discount for long race horizon
    gae_lambda: float = Field(default=0.95)
    clip_range: float = Field(default=0.2)
    ent_coef: float = Field(default=0.01)  # small entropy bonus for exploration
    min_drivers_per_race: int = Field(default=10)
    pit_loss_s: float = Field(default=25.0)
    noise_std: float = Field(default=0.2)

    mlflow_tracking_uri: str = Field(
        default="mlruns", validation_alias="PITWALL_MLFLOW_TRACKING_URI"
    )
    mlflow_experiment: str = Field(default="strategy-policy")
    registered_model_name: str = Field(default="pitwall-strategy-policy")

    tire_model_uri: str = Field(
        default="models:/pitwall-tire-degradation/latest",
        validation_alias="PITWALL_TIRE_MODEL_URI",
    )
    sc_model_uri: str = Field(
        default="models:/pitwall-safety-car/latest",
        validation_alias="PITWALL_SC_MODEL_URI",
    )


# ---------------------------------------------------------------------------
# Multi-race environment wrapper
# ---------------------------------------------------------------------------


class MultiRaceEnv(gym.Env[np.ndarray, np.ndarray]):
    """F1RaceEnv wrapper that samples a random (race, driver) pair on each reset.

    Sampling diverse races and driver baselines is the primary regulariser
    preventing the policy from over-fitting to a single circuit layout or car
    performance level.
    """

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
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        idx = int(self.np_random.integers(0, len(self._pool)))
        race_df, driver = self._pool[idx]
        return self._inner.reset(
            seed=seed,
            options={"race_df": race_df, "driver_number": driver},
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:  # type: ignore[override]
        return self._inner.step(action)


# ---------------------------------------------------------------------------
# Race pool
# ---------------------------------------------------------------------------


def build_race_pool(
    gold_df: pd.DataFrame,
    min_drivers: int = 10,
) -> list[tuple[pd.DataFrame, Any]]:
    """Build (race_df, driver_number) pairs for multi-race PPO training.

    Args:
        gold_df: Full gold DataFrame across one or more seasons.
        min_drivers: Minimum drivers with ≥10 complete laps for a race to qualify.

    Returns:
        List of (race_df, driver_number) tuples — one entry per controllable driver
        per qualifying race.
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
        # One copy per race shared by all drivers — avoids 20× duplication.
        shared_race_df = race_df.copy()
        for drv in eligible:
            pool.append((shared_race_df, drv))
        n_races += 1

    logger.info("Race pool: %d (race, driver) pairs across %d races.", len(pool), n_races)
    return pool


# ---------------------------------------------------------------------------
# MLflow callback
# ---------------------------------------------------------------------------


class _MLflowCallback(BaseCallback):
    """Log mean episode reward to MLflow every ``log_freq`` timesteps."""

    def __init__(self, log_freq: int = 50_000) -> None:
        super().__init__()
        self._log_freq = log_freq
        self._last_logged = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_logged >= self._log_freq:
            ep_rewards = [
                info["episode"]["r"] for info in self.locals.get("infos", []) if "episode" in info
            ]
            if ep_rewards:
                mlflow.log_metric(
                    "mean_episode_reward",
                    float(np.mean(ep_rewards)),
                    step=self.num_timesteps,
                )
            self._last_logged = self.num_timesteps
        return True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(config: StrategyPolicyConfig | None = None) -> None:
    """Train the PPO strategy policy and register it in MLflow.

    Args:
        config: Training configuration (defaults to env-vars / built-in defaults).
    """
    if config is None:
        config = StrategyPolicyConfig()

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

    def _make_env() -> gym.Env[np.ndarray, np.ndarray]:
        return MultiRaceEnv(
            race_pool,
            tire_model,
            sc_model,
            config.pit_loss_s,
            config.noise_std,
        )

    vec_env = DummyVecEnv([_make_env for _ in range(config.n_envs)])

    ppo_hparams: dict[str, Any] = {
        "n_steps": config.n_steps,
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "learning_rate": config.learning_rate,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "clip_range": config.clip_range,
        "ent_coef": config.ent_coef,
    }

    with mlflow.start_run():
        mlflow.log_params(
            {
                **ppo_hparams,
                "net_arch": "128x64",
                "device": "cpu",
                "training_seasons": config.training_seasons,
                "n_envs": config.n_envs,
                "total_timesteps": config.total_timesteps,
                "race_pool_size": len(race_pool),
                "pit_loss_s": config.pit_loss_s,
                "obs_dim": 20,
            }
        )

        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            policy_kwargs={"net_arch": [128, 64]},  # wider first layer for 20-dim obs
            device="cpu",
            verbose=1,
            **ppo_hparams,
        )

        logger.info(
            "Training PPO on device: %s | %d timesteps across %d envs",
            model.device,
            config.total_timesteps,
            config.n_envs,
        )

        model.learn(
            total_timesteps=config.total_timesteps,
            callback=_MLflowCallback(log_freq=50_000),
            progress_bar=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            policy_path = os.path.join(tmpdir, "policy")
            model.save(policy_path)  # writes policy.zip
            mlflow.log_artifact(f"{policy_path}.zip", artifact_path="model")
            mlflow.set_tag("registered_model_name", config.registered_model_name)

        logger.info(
            "PPO training complete — model logged to experiment '%s'.",
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
        typer.Option("--timesteps", help="Total PPO training timesteps."),
    ] = 1_000_000,
    n_envs: Annotated[
        int,
        typer.Option("--n-envs", help="Number of parallel environments."),
    ] = 4,
) -> None:
    """CLI entry point for PPO strategy policy training."""
    season_list = [int(s.strip()) for s in seasons.split(",")]
    train(
        StrategyPolicyConfig(
            training_seasons=season_list,
            total_timesteps=timesteps,
            n_envs=n_envs,
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
    ] = 500_000,
    seasons: Annotated[
        str,
        typer.Option("--seasons", help="Comma-separated seasons."),
    ] = "2024",
    n_envs: Annotated[
        int,
        typer.Option("--n-envs", help="Number of parallel environments."),
    ] = 4,
) -> None:
    """Fine-tune an existing PPO checkpoint with the current reward function."""
    config = StrategyPolicyConfig(
        training_seasons=[int(s.strip()) for s in seasons.split(",")],
        total_timesteps=timesteps,
        n_envs=n_envs,
    )

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment)

    logger.info("Loading tire and SC models from MLflow registry...")
    tire_model = mlflow.pyfunc.load_model(config.tire_model_uri)
    sc_model = mlflow.pyfunc.load_model(config.sc_model_uri)

    gold_df = load_gold_seasons(config.training_seasons)
    race_pool = build_race_pool(gold_df, config.min_drivers_per_race)

    def _make_env() -> gym.Env[np.ndarray, np.ndarray]:
        return MultiRaceEnv(race_pool, tire_model, sc_model, config.pit_loss_s, config.noise_std)

    vec_env = DummyVecEnv([_make_env for _ in range(config.n_envs)])

    logger.info("Loading checkpoint: %s", checkpoint)
    model = PPO.load(checkpoint, env=vec_env, device="cpu")

    logger.info("Fine-tuning for %d additional timesteps...", timesteps)
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
