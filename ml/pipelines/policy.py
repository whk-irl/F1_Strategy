"""Historical action pretraining, history-aware PPO and paired policy evaluation."""

from __future__ import annotations

import copy
import logging
from collections import deque
from functools import cache
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from services.ingestion.acquire import write_json
from services.simulator.env import F1RaceEnv, _synthetic_deg
from sklearn.metrics import balanced_accuracy_score
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.data import DataLoader, TensorDataset

from ml.models.strategy_policy.train import build_race_pool
from ml.pipelines.config import TrainingConfig

LOGGER = logging.getLogger(__name__)


def dry_races(df: pd.DataFrame) -> pd.DataFrame:
    """Select complete dry events: wet compound actions are outside Discrete(4)."""
    wet = (
        df["compound_encoded"]
        .isin([3, 4])
        .groupby([df["year"], df["round_number"]])
        .transform("any")
    )
    red = (
        df["track_status_encoded"].eq(4).groupby([df["year"], df["round_number"]]).transform("any")
    )
    return df[~wet & ~red].copy()


def historical_actions(driver: pd.DataFrame) -> dict[int, int]:
    """Label pit-in laps with the next recorded compound; future data is labels only."""
    rows = driver.sort_values("lap_number").reset_index(drop=True)
    actions = dict.fromkeys(rows["lap_number"].astype(int).tolist(), 0)
    for idx in range(len(rows)):
        row = rows.iloc[idx]
        if bool(row["pit_in_this_lap"]) and idx + 1 < len(rows):
            following = rows.iloc[idx + 1]
            compound = int(following["compound_encoded"])
            if compound in (0, 1, 2) and following["lap_number"] == row["lap_number"] + 1:
                actions[int(row["lap_number"])] = compound + 1
    return actions


class HistoryEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """Stack lap histories plus observable compound-use flags, zeroing on reset."""

    def __init__(self, env: F1RaceEnv, frames: int) -> None:
        super().__init__(env)
        self.simulator = env
        self.frames = frames
        self.width = 24
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.width * frames,), dtype=np.float32
        )
        self.history: deque[np.ndarray] = deque(maxlen=frames)

    def _append(self, obs: np.ndarray) -> np.ndarray:
        flags = np.array(
            [float(i in self.simulator._used_compounds) for i in range(3)], dtype=np.float32
        )
        self.history.append(np.concatenate([obs, flags]))
        return np.concatenate(list(self.history)).astype(np.float32)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Clear history so no preceding driver or race can leak into this episode."""
        obs, info = self.env.reset(**kwargs)
        self.history.clear()
        self.history.extend(np.zeros(self.width, dtype=np.float32) for _ in range(self.frames - 1))
        return self._append(obs), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the simulator and expose a dry-compound compliance signal."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        legal = len(self.simulator._used_compounds & {0, 1, 2}) >= 2
        reward = float(reward)
        if terminated and not legal:
            reward -= 50.0
        info["compound_compliant"] = legal
        info["stops"] = self.simulator._stops_taken
        return self._append(obs), reward, terminated, truncated, info


def demonstrations(df: pd.DataFrame, tire: Any, sc: Any, config: TrainingConfig) -> TensorDataset:
    """Replay historical actions to generate observations in the training schema."""
    observations, actions = [], []
    pool = build_race_pool(dry_races(df))
    for race, driver in pool:
        driver_rows = race[race["driver_number"].eq(driver)]
        if int(driver_rows["lap_number"].max()) < int(race["lap_number"].max()):
            continue
        labels = historical_actions(driver_rows)
        env = HistoryEnv(
            F1RaceEnv(race, driver, tire, sc, noise_std=0.0, replay_mode=True),
            config.history_frames,
        )
        obs, _ = env.reset(seed=config.seed)
        while True:
            lap = env.simulator._lap
            if lap not in labels:
                break
            action = labels[lap]
            # The first lap's recorded position is also the legacy starting-position estimate.
            if lap > 1:
                observations.append(obs.copy())
                actions.append(action)
            obs, _, done, truncated, _ = env.step(action)
            if done or truncated:
                break
        env.close()
    if not actions or len(set(actions)) < 2:
        raise ValueError("Need historical stay and pit decisions for behavioral cloning")
    LOGGER.info(
        "Built %d demonstrations (%d pit decisions)", len(actions), np.count_nonzero(actions)
    )
    return TensorDataset(
        torch.from_numpy(np.asarray(observations, dtype=np.float32)),
        torch.tensor(actions, dtype=torch.long),
    )


def pretrain_behavior(
    model: PPO, train: TensorDataset, val: TensorDataset, config: TrainingConfig
) -> dict[str, Any]:
    """Fit policy logits using capped inverse-frequency class weights and early stopping."""
    _, y = train.tensors
    counts = torch.bincount(y, minlength=4).float()
    # Square-root balancing improves rare pit exposure without turning half of all laps into pits.
    weights = (counts.sum() / counts.clamp(min=1)).sqrt().clamp(max=8)
    weights /= weights.mean()
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=config.learning_rate)
    loader = DataLoader(
        train,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    best, stale, best_weights = float("inf"), 0, copy.deepcopy(model.policy.state_dict())
    history = []
    for epoch in range(config.bc_epochs):
        model.policy.set_training_mode(True)
        for obs, labels in loader:
            _, log_prob, _ = model.policy.evaluate_actions(
                obs.to(model.device), labels.to(model.device)
            )
            loss = -(log_prob * weights.to(model.device)[labels.to(model.device)]).mean()
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            optimizer.step()
        model.policy.set_training_mode(False)
        with torch.no_grad():
            vx, vy = val.tensors
            _, log_prob, _ = model.policy.evaluate_actions(vx.to(model.device), vy.to(model.device))
            val_loss = float(-(log_prob * weights.to(model.device)[vy.to(model.device)]).mean())
            predicted, _ = model.predict(vx.numpy(), deterministic=True)
        balanced = float(balanced_accuracy_score(vy.numpy(), predicted))
        history.append(
            {"epoch": epoch + 1, "val_weighted_nll": val_loss, "val_balanced_accuracy": balanced}
        )
        LOGGER.info(
            "Behavior pretraining epoch %d: NLL %.4f, balanced accuracy %.3f",
            epoch + 1,
            val_loss,
            balanced,
        )
        if val_loss < best:
            best, stale, best_weights = val_loss, 0, copy.deepcopy(model.policy.state_dict())
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.policy.load_state_dict(best_weights)
    model.save(str(config.output_dir / "policy" / "behavior"))
    return {
        "training_rows": len(train),
        "validation_rows": len(val),
        "class_counts": counts.tolist(),
        "history": history,
    }


class SampledRaceEnv(gym.Env[np.ndarray, int]):
    """Reuse a simulator, sampling only explicitly supplied training events."""

    def __init__(
        self, pool: list[tuple[pd.DataFrame, Any]], tire: Any, sc: Any, frames: int
    ) -> None:
        super().__init__()
        if not pool:
            raise ValueError("No eligible dry race/driver pairs")
        self.pool = pool
        self.inner = HistoryEnv(F1RaceEnv(*pool[0], tire, sc), frames)
        self.observation_space, self.action_space = (
            self.inner.observation_space,
            self.inner.action_space,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Sample a race with Gymnasium's seeded RNG."""
        super().reset(seed=seed)
        race, driver = self.pool[int(self.np_random.integers(len(self.pool)))]
        return self.inner.reset(seed=seed, options={"race_df": race, "driver_number": driver})

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Forward one action to the current driver."""
        return self.inner.step(action)

    def close(self) -> None:
        """Close the inner environment."""
        self.inner.close()


def make_env(df: pd.DataFrame, tire: Any, sc: Any, config: TrainingConfig) -> DummyVecEnv:
    """Build independent seeded history buffers for each rollout environment."""
    pool = build_race_pool(dry_races(df))
    env = DummyVecEnv(
        [
            lambda: Monitor(SampledRaceEnv(pool, tire, sc, config.history_frames))
            for _ in range(config.n_envs)
        ]
    )
    env.seed(config.seed)
    return env


class DynamicProgramming:
    """Deterministic minimum-time baseline with two-compound state and no future flags."""

    def __init__(self, env: F1RaceEnv) -> None:
        self.env = env

    @cache  # noqa: B019 -- explicitly cleared after each evaluation episode
    def value(self, remaining: int, age: int, compound: int, mask: int) -> tuple[float, int]:
        """Compute nominal remaining degradation plus end-of-lap pit costs."""
        if remaining == 0:
            return (0.0 if mask.bit_count() >= 2 else 1e6), 0
        running = _synthetic_deg(compound, age, self.env._degrade_rate, self.env._cliff_lap)
        future, _ = self.value(remaining - 1, age + 1, compound, mask)
        best = (running + future, 0)
        if remaining > 1:
            for target in range(3):
                future, _ = self.value(remaining - 1, 1, target, mask | (1 << target))
                candidate = (running + self.env._pit_loss_s + future, target + 1)
                if candidate[0] < best[0]:
                    best = candidate
        return best

    def action(self) -> int:
        """Replan from current age and compound without seeing future race events."""
        env = self.env
        mask = sum(1 << i for i in env._used_compounds if i in (0, 1, 2))
        return self.value(env._total_laps - env._lap + 1, env._tyre_life, env._compound, mask)[1]


def evaluate(
    model: PPO | None,
    df: pd.DataFrame,
    tire: Any,
    sc: Any,
    config: TrainingConfig,
    baseline: str | None = None,
) -> dict[str, Any]:
    """Use fixed paired race/driver/seed episodes, reporting real outcome metrics."""
    pool = build_race_pool(dry_races(df))
    if not pool:
        raise ValueError("No eligible dry evaluation races")
    # Interleave drivers across events before selecting episodes, avoiding one-track evaluations.
    groups: dict[tuple[int, int], list[tuple[pd.DataFrame, Any]]] = {}
    for race, driver in pool:
        key = (int(race["year"].iloc[0]), int(race["round_number"].iloc[0]))
        groups.setdefault(key, []).append((race, driver))
    paired = [
        items[i]
        for i in range(max(map(len, groups.values())))
        for items in groups.values()
        if i < len(items)
    ][: config.eval_episodes]
    records = []
    for index, (race, driver) in enumerate(paired):
        inner = F1RaceEnv(race, driver, tire, sc, noise_std=0.2)
        env = HistoryEnv(inner, config.history_frames)
        obs, _ = env.reset(seed=config.seed + index)
        planner = DynamicProgramming(inner) if baseline == "dynamic_programming" else None
        total_reward, total_time = 0.0, 0.0
        while True:
            if planner is not None:
                action = planner.action()
            elif baseline == "tyre_age":
                action = (
                    3
                    if inner._tyre_life >= inner._cliff_lap[inner._compound] - 3
                    and inner._lap < inner._total_laps - 6
                    else 0
                )
                if action == 3 and inner._compound == 2:
                    action = 2
            elif model is not None:
                action = int(model.predict(obs, deterministic=True)[0])
            else:
                action = 0
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            total_time += info["sim_lap_time_s"]
            if done or truncated:
                break
        records.append(
            {
                "year": int(race["year"].iloc[0]),
                "round": int(race["round_number"].iloc[0]),
                "driver": str(driver),
                "seed": config.seed + index,
                "reward": total_reward,
                "time_s": total_time,
                "position": info["position"],
                "stops": info["stops"],
                "compound_compliant": info["compound_compliant"],
            }
        )
        env.close()
        if planner is not None:
            planner.value.cache_clear()
    return {
        "episodes": records,
        "mean_reward": float(np.mean([x["reward"] for x in records])),
        "mean_position": float(np.mean([x["position"] for x in records])),
        "mean_time_s": float(np.mean([x["time_s"] for x in records])),
        "mean_stops": float(np.mean([x["stops"] for x in records])),
        "compound_compliance": float(np.mean([x["compound_compliant"] for x in records])),
    }


class ValidationCheckpoint(BaseCallback):
    """Select policy checkpoints using validation events only, including the initial policy."""

    def __init__(
        self, df: pd.DataFrame, tire: Any, sc: Any, config: TrainingConfig, path: Path
    ) -> None:
        super().__init__()
        self.df, self.tire, self.sc, self.config, self.path = df, tire, sc, config, path
        self.best = -float("inf")
        self.last = 0
        self.history: list[dict[str, Any]] = []

    def check(self) -> None:
        """Persist an improved mean validation reward and its training step."""
        assert isinstance(self.model, PPO)
        result = evaluate(self.model, self.df, self.tire, self.sc, self.config)
        score = result["mean_reward"]
        self.history.append({"timesteps": self.model.num_timesteps, "mean_reward": score})
        LOGGER.info("Policy validation step %d: reward %.3f", self.model.num_timesteps, score)
        if score > self.best:
            self.best = score
            self.model.save(str(self.path))
        write_json(self.path.with_suffix(".history.json"), self.history)

    def _on_training_start(self) -> None:
        self.last = self.num_timesteps
        self.check()

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last >= self.config.eval_frequency:
            self.check()
            self.last = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        self.check()


def fit_policy(
    splits: dict[str, pd.DataFrame], tire: Any, sc: Any, config: TrainingConfig
) -> dict[str, Any]:
    """Run BC → source PPO → newer-event fine-tuning and evaluate once on test events."""
    path = config.output_dir / "policy"
    path.mkdir(parents=True, exist_ok=True)
    env = make_env(splits["pretrain"], tire, sc, config)
    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        seed=config.seed,
        n_steps=config.rollout_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        gamma=0.995,
        ent_coef=0.005,
        target_kl=0.025,
        n_epochs=5,
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
    )
    behavior = pretrain_behavior(
        model,
        demonstrations(splits["pretrain"], tire, sc, config),
        demonstrations(splits["pretrain_val"], tire, sc, config),
        config,
    )
    write_json(path / "behavior.metrics.json", behavior)
    source_callback = ValidationCheckpoint(
        splits["pretrain_val"], tire, sc, config, path / "pretrained"
    )
    model.learn(config.pretrain_steps, callback=source_callback)
    model.save(str(path / "pretrained_last"))
    env.close()
    env = make_env(splits["finetune"], tire, sc, config)
    model = PPO.load(
        str(path / "pretrained"), env=env, device="cpu", learning_rate=config.finetune_learning_rate
    )
    fine_callback = ValidationCheckpoint(
        splits["finetune_val"], tire, sc, config, path / "finetuned"
    )
    model.learn(config.finetune_steps, callback=fine_callback, reset_num_timesteps=False)
    model.save(str(path / "finetuned_last"))
    env.close()
    result: dict[str, Any] = {
        "behavior": behavior,
        "pretraining_validation": source_callback.history,
        "finetuning_validation": fine_callback.history,
        "test": {},
    }
    for name in ("behavior", "pretrained", "finetuned"):
        candidate = PPO.load(str(path / name), device="cpu")
        result["test"][name] = evaluate(candidate, splits["test"], tire, sc, config)
    for name in ("tyre_age", "dynamic_programming"):
        result["test"][name] = evaluate(None, splits["test"], tire, sc, config, baseline=name)
    return result
