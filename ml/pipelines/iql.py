"""Temporal Implicit Q-Learning for supported-action race strategy.

The learner never maximizes over an action that is absent from the logged
transition.  A value expectile and advantage-weighted behavior objective make
the policy conservative enough for historical race data while a causal
Transformer retains the preceding tyre/traffic context.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from services.ingestion.acquire import write_json
from services.simulator.env import F1RaceEnv
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.models.strategy_policy.train import build_race_pool
from ml.pipelines.config import TrainingConfig
from ml.pipelines.policy import HistoryEnv, dry_races, evaluate, historical_actions

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransitionBatch:
    """Dense offline transition arrays."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray

    def tensors(self) -> TensorDataset:
        """Convert arrays to a shuffled-loader compatible dataset."""
        return TensorDataset(
            torch.from_numpy(self.observations),
            torch.from_numpy(self.actions),
            torch.from_numpy(self.rewards),
            torch.from_numpy(self.next_observations),
            torch.from_numpy(self.terminals),
        )


def collect_historical_transitions(
    frame: pd.DataFrame, tire: Any, sc: Any, config: TrainingConfig
) -> TransitionBatch:
    """Replay complete real driver races into causal state-action transitions."""
    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    next_observations: list[np.ndarray] = []
    terminals: list[float] = []
    for race, driver in build_race_pool(dry_races(frame)):
        rows = race[race["driver_number"].eq(driver)].sort_values("lap_number")
        if rows.empty or int(rows["lap_number"].max()) < int(race["lap_number"].max()):
            continue
        labels = historical_actions(rows)
        env = HistoryEnv(
            F1RaceEnv(race, driver, tire, sc, noise_std=0.0, replay_mode=True),
            config.history_frames,
        )
        observation, _ = env.reset(seed=config.seed)
        while True:
            action = labels.get(env.simulator._lap)
            if action is None:
                break
            following, reward, done, truncated, _ = env.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(following)
            terminals.append(float(done or truncated))
            observation = following
            if done or truncated:
                break
        env.close()
    if not actions or len(set(actions)) < 2:
        raise ValueError("Historical transitions need both stay-out and pit actions")
    LOGGER.info(
        "Collected %d transitions from %d events (%d pit actions)",
        len(actions),
        frame[["year", "round_number"]].drop_duplicates().shape[0],
        np.count_nonzero(actions),
    )
    return TransitionBatch(
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
        np.asarray(rewards, dtype=np.float32),
        np.asarray(next_observations, dtype=np.float32),
        np.asarray(terminals, dtype=np.float32),
    )


class TemporalEncoder(nn.Module):
    """Causal Transformer encoder over lap observations."""

    def __init__(self, frames: int, width: int, hidden: int, layers: int, heads: int) -> None:
        super().__init__()
        self.frames, self.width = frames, width
        self.project = nn.Linear(width, hidden)
        self.position = nn.Parameter(torch.zeros(1, frames, hidden))
        block = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 4, dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode only current and preceding frames."""
        sequence = observation.reshape(-1, self.frames, self.width)
        encoded = self.project(sequence) + self.position
        mask = torch.triu(
            torch.ones(self.frames, self.frames, device=observation.device, dtype=torch.bool),
            diagonal=1,
        )
        return cast(torch.Tensor, self.norm(self.transformer(encoded, mask=mask)[:, -1]))


class TemporalHead(nn.Module):
    """Temporal encoder with either a scalar or discrete-action output."""

    def __init__(self, config: TrainingConfig, outputs: int) -> None:
        super().__init__()
        self.encoder = TemporalEncoder(
            config.history_frames,
            24,
            config.iql_hidden_dim,
            config.iql_layers,
            config.iql_heads,
        )
        self.head = nn.Sequential(
            nn.Linear(config.iql_hidden_dim, config.iql_hidden_dim),
            nn.GELU(),
            nn.Linear(config.iql_hidden_dim, outputs),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.head(self.encoder(observation)))


class TemporalIQLPolicy:
    """Serializable actor plus offline IQL critics used during training."""

    def __init__(self, config: TrainingConfig, mean: np.ndarray, std: np.ndarray) -> None:
        self.config = config
        requested = (
            "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
        )
        self.device = torch.device("cpu" if requested == "auto" else requested)
        self.mean = torch.as_tensor(mean, dtype=torch.float32, device=self.device)
        self.std = torch.as_tensor(std, dtype=torch.float32, device=self.device)
        self.q1 = TemporalHead(config, 4).to(self.device)
        self.q2 = TemporalHead(config, 4).to(self.device)
        self.value = TemporalHead(config, 1).to(self.device)
        self.actor = TemporalHead(config, 4).to(self.device)

    def _normalize(self, observation: torch.Tensor) -> torch.Tensor:
        return (observation.to(self.device) - self.mean) / self.std

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        """Select an action and enforce the mandatory two-compound race constraint."""
        array = np.asarray(observation, dtype=np.float32)
        single = array.ndim == 1
        tensor = torch.from_numpy(array[None] if single else array)
        self.actor.eval()
        with torch.no_grad():
            logits = self.actor(self._normalize(tensor))
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = torch.distributions.Categorical(logits=logits).sample()  # type: ignore[no-untyped-call]
            raw = tensor.reshape(-1, self.config.history_frames, 24)[:, -1]
            for index in range(len(action)):
                used = raw[index, 21:24] > 0.5
                # Preserve enough laps to make the required compound change. This is a
                # sporting-rule shield; IQL still decides the earlier strategic window.
                if raw[index, 1] <= 0.15 and int(used.sum()) < 2:
                    candidates = (~used).nonzero().flatten() + 1
                    action[index] = candidates[logits[index, candidates].argmax()]
                # Reject an immediate second stop on a nearly new set.
                elif action[index] > 0 and raw[index, 2] < (2 / 60):
                    action[index] = 0
        result = action.cpu().numpy()
        return (result[0] if single else result), None

    def save(self, path: Path) -> None:
        """Save inference and critic weights with their observation normalization."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config.model_dump(mode="json"),
                "mean": self.mean.cpu(),
                "std": self.std.cpu(),
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "value": self.value.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> TemporalIQLPolicy:
        """Restore a policy checkpoint on an explicit serving device."""
        payload = torch.load(path, map_location=device, weights_only=True)
        config = TrainingConfig.model_validate(payload["config"]).model_copy(
            update={"device": device}
        )
        policy = cls(config, payload["mean"].numpy(), payload["std"].numpy())
        for name in ("actor", "q1", "q2", "value"):
            getattr(policy, name).load_state_dict(payload[name])
        return policy


def _expectile_loss(difference: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(difference > 0, expectile, 1 - expectile)
    return (weight * difference.square()).mean()


def fit_iql_stage(
    policy: TemporalIQLPolicy,
    transitions: TransitionBatch,
    validation: pd.DataFrame,
    tire: Any,
    sc: Any,
    config: TrainingConfig,
    *,
    epochs: int,
    learning_rate: float,
    checkpoint: Path,
) -> list[dict[str, float]]:
    """Optimize Q, expectile value, and advantage-weighted behavior objectives."""
    modules = [policy.q1, policy.q2, policy.value, policy.actor]
    optimizer = torch.optim.AdamW(
        [parameter for module in modules for parameter in module.parameters()],
        lr=learning_rate,
        weight_decay=1e-4,
    )
    loader = DataLoader(
        transitions.tensors(),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        pin_memory=policy.device.type == "cuda",
    )
    counts = np.bincount(transitions.actions, minlength=4).astype(np.float32)
    action_weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    action_weights = np.clip(action_weights / action_weights.mean(), 0.25, 8.0)
    action_weights_tensor = torch.as_tensor(action_weights, device=policy.device)
    best_reward = -float("inf")
    best = copy.deepcopy(
        {name: getattr(policy, name).state_dict() for name in ("actor", "q1", "q2", "value")}
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        totals = np.zeros(4, dtype=float)
        batches = 0
        for obs, action, reward, next_obs, terminal in loader:
            obs = policy._normalize(obs)
            next_obs = policy._normalize(next_obs)
            action = action.to(policy.device)
            reward = reward.to(policy.device)
            terminal = terminal.to(policy.device)
            q1_all, q2_all = policy.q1(obs), policy.q2(obs)
            q1 = q1_all.gather(1, action[:, None]).squeeze(1)
            q2 = q2_all.gather(1, action[:, None]).squeeze(1)
            with torch.no_grad():
                minimum_q = torch.minimum(q1, q2)
            value = policy.value(obs).squeeze(1)
            value_loss = _expectile_loss(minimum_q - value, config.iql_expectile)
            with torch.no_grad():
                target = reward + config.iql_discount * (1 - terminal) * policy.value(
                    next_obs
                ).squeeze(1)
            q_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
            with torch.no_grad():
                advantage = minimum_q - value
                weights = torch.exp(advantage / config.iql_temperature).clamp(max=100.0)
            actor_loss = (
                nn.functional.cross_entropy(policy.actor(obs), action, reduction="none")
                * weights
                * action_weights_tensor[action]
            ).mean()
            loss = q_loss + value_loss + actor_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(
                [parameter for module in modules for parameter in module.parameters()], 1.0
            )
            optimizer.step()
            totals += [
                float(loss.item()),
                float(q_loss.item()),
                float(value_loss.item()),
                float(actor_loss.item()),
            ]
            batches += 1
        record = {
            "epoch": float(epoch),
            "loss": totals[0] / batches,
            "q_loss": totals[1] / batches,
            "value_loss": totals[2] / batches,
            "actor_loss": totals[3] / batches,
        }
        if epoch % config.iql_eval_interval == 0 or epoch == epochs:
            result = evaluate(policy, validation, tire, sc, config)
            record["validation_reward"] = result["mean_reward"]
            record["validation_time_s"] = result["mean_time_s"]
            if result["mean_reward"] > best_reward:
                best_reward = result["mean_reward"]
                best = copy.deepcopy(
                    {
                        name: getattr(policy, name).state_dict()
                        for name in ("actor", "q1", "q2", "value")
                    }
                )
                policy.save(checkpoint)
        history.append(record)
        LOGGER.info("IQL epoch %d/%d: loss %.4f", epoch, epochs, record["loss"])
    for name, weights in best.items():
        getattr(policy, name).load_state_dict(weights)
    policy.save(checkpoint)
    write_json(checkpoint.with_suffix(".history.json"), history)
    return history


def fit_temporal_iql(
    splits: dict[str, pd.DataFrame], tire: Any, sc: Any, config: TrainingConfig
) -> dict[str, Any]:
    """Pretrain on older races, fine-tune on recent races, then open the test set once."""
    target = config.output_dir / "policy_iql"
    source = collect_historical_transitions(splits["pretrain"], tire, sc, config)
    mean = source.observations.mean(axis=0)
    std = source.observations.std(axis=0).clip(min=1e-6)
    policy = TemporalIQLPolicy(config, mean, std)
    pretrain = fit_iql_stage(
        policy,
        source,
        splits["pretrain_val"],
        tire,
        sc,
        config,
        epochs=config.iql_epochs,
        learning_rate=config.learning_rate,
        checkpoint=target / "pretrained.pt",
    )
    recent = collect_historical_transitions(splits["finetune"], tire, sc, config)
    finetune = fit_iql_stage(
        policy,
        recent,
        splits["finetune_val"],
        tire,
        sc,
        config,
        epochs=config.iql_finetune_epochs,
        learning_rate=config.finetune_learning_rate,
        checkpoint=target / "finetuned.pt",
    )
    result = {
        "algorithm": "causal-transformer discrete implicit Q-learning",
        "source_transitions": len(source.actions),
        "source_pit_actions": int(np.count_nonzero(source.actions)),
        "recent_transitions": len(recent.actions),
        "recent_pit_actions": int(np.count_nonzero(recent.actions)),
        "pretrain_history": pretrain,
        "finetune_history": finetune,
        "test": {
            "iql": evaluate(policy, splits["test"], tire, sc, config),
            "dynamic_programming": evaluate(
                None, splits["test"], tire, sc, config, baseline="dynamic_programming"
            ),
            "tyre_age": evaluate(None, splits["test"], tire, sc, config, baseline="tyre_age"),
        },
    }
    write_json(target / "metrics.json", result)
    return result
