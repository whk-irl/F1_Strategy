#!/usr/bin/env bash
# Submit only after the 2021 acquisition job has completed successfully.
#SBATCH --job-name=f1-iql-seeds
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"
mkdir -p logs

SEEDS=(11 23 42 67 89)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1

uv sync --frozen
uv run python -m ml.pipelines.train_iql \
  --config configs/training.iql-hpc.json \
  --seed "$SEED" \
  --output-dir "runs/temporal-iql-2021-2026-seed-${SEED}" \
  --resume
