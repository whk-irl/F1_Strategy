#!/usr/bin/env bash
#SBATCH --job-name=f1-temporal-iql
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"
mkdir -p logs .fastf1_cache

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1

uv sync --frozen
uv run python -m services.ingestion.acquire --years 2021 --provider fastf1
uv run python -m ml.pipelines.pit_time
uv run python -m ml.pipelines.train_iql \
  --config configs/training.iql-hpc.json \
  --resume
