#!/usr/bin/env bash
# PPO pit policy on the calibrated race-time model (ml/strategy). One seed per array task.
# CPU-only: 2x128 MLP, numpy env, envs stepped in one process (DummyVecEnv), so 4 CPUs suffice.
# Submit from the bundle/project root:
#   module load slurm/slurm/21.08.8; mkdir -p logs
#   PYTHON=/path/to/.venv/bin/python sbatch --export=ALL infra/slurm/train_strategy_ppo.sbatch
# Optional: TIMESTEPS=40000000. Outputs runs/strategy-ppo-<jobid>/seed-<seed>/benchmark.json
# No --mem: this cluster's defq advertises RealMemory=1 MB and rejects memory requests.
#SBATCH --job-name=f1-strategy-ppo
#SBATCH --partition=defq
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
PY="${PYTHON:-.venv/bin/python}"

seeds=(42 11 23 67 89)
seed="${seeds[${SLURM_ARRAY_TASK_ID:?Missing array index}]}"
timesteps="${TIMESTEPS:-20000000}"
out="runs/strategy-ppo-${SLURM_ARRAY_JOB_ID}/seed-${seed}"
test -f data/gold_2022_2025.parquet
test ! -e "$out"

"$PY" -c "import torch, gymnasium, stable_baselines3, pyarrow, pandas, pytest; print('python ok', torch.__version__, stable_baselines3.__version__)"
# Override repo addopts: its --cov-fail-under=80 gate is for the full suite, not one file.
"$PY" -m pytest -q -o addopts="" -p no:cacheprovider tests/unit/test_strategy_planner.py
"$PY" -m ml.strategy.train_ppo \
    --data data/gold_2022_2025.parquet \
    --eval-years 2024,2025 \
    --timesteps "$timesteps" \
    --n-envs 16 \
    --seed "$seed" \
    --device cpu \
    --out "$out"
