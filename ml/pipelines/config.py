"""Validated, portable configuration for the full training lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainingConfig(BaseModel):
    """Event splits and resource budgets; no test event is used for selection."""

    model_config = ConfigDict(extra="forbid")

    gold_path: Path = Path("data/gold_2022_2025.parquet")
    lake_root: Path = Path("data/lake")
    output_dir: Path = Path("runs/staged")
    pretrain_years: list[int] = Field(default_factory=lambda: [2022, 2023])
    finetune_years: list[int] = Field(default_factory=lambda: [2025, 2026])
    gate_years: list[int] = Field(default_factory=lambda: [2024])
    validation_rounds: list[int] = Field(default_factory=lambda: [5, 10, 15])
    finetune_validation_start: tuple[int, int] = (2026, 11)
    test_start: tuple[int, int] = (2026, 14)
    seed: int = 42
    history_frames: int = Field(default=4, ge=1, le=16)
    bc_epochs: int = Field(default=30, ge=1)
    sequence_epochs: int = Field(default=40, ge=1)
    sequence_finetune_epochs: int = Field(default=15, ge=1)
    sequence_length: int = Field(default=10, ge=2)
    patience: int = Field(default=7, ge=1)
    batch_size: int = Field(default=256, ge=2)
    pretrain_steps: int = Field(default=200_000, ge=1)
    finetune_steps: int = Field(default=100_000, ge=1)
    n_envs: int = Field(default=4, ge=1)
    rollout_steps: int = Field(default=512, ge=2)
    eval_episodes: int = Field(default=40, ge=1)
    eval_frequency: int = Field(default=25_000, ge=1)
    min_spearman: float = Field(default=0.7, ge=0.7, le=1.0)
    learning_rate: float = Field(default=3e-4, gt=0)
    finetune_learning_rate: float = Field(default=5e-5, gt=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    tracking_uri: str | None = "sqlite:///runs/mlflow.db"

    @model_validator(mode="after")
    def check_splits(self) -> TrainingConfig:
        """Reject overlapping years and holdout rounds before any training."""
        groups = [set(self.pretrain_years), set(self.finetune_years), set(self.gate_years)]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError("Pretraining, fine-tuning and simulator-gate years must be disjoint")
        if self.finetune_validation_start >= self.test_start:
            raise ValueError("Fine-tuning validation must precede test events")
        if self.rollout_steps * self.n_envs % self.batch_size:
            raise ValueError("batch_size must divide rollout_steps * n_envs")
        return self

    @classmethod
    def load(cls, path: Path) -> TrainingConfig:
        """Read a JSON experiment configuration."""
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
