#!/usr/bin/env bash
# P2 training loop: 4 arms x 5 seeds = 20 tasks. Task i -> arm ARMS[i/5], seed SEEDS[i%5].
# CPU-only (MLP + numpy env, envs stepped in one process). No --mem: defq rejects it.
#SBATCH --job-name=f1-loop-train
#SBATCH --partition=defq
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the bundle root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PY="${PYTHON:?export PYTHON}"
ARMS=(A B C D); SEEDS=(42 11 23 67 89)
i="${SLURM_ARRAY_TASK_ID:?}"
arm="${ARMS[$((i / 5))]}"; seed="${SEEDS[$((i % 5))]}"
"$PY" -m ml.strategy.train_ppo --arm "$arm" --seed "$seed" \
  --timesteps "${TIMESTEPS:-20000000}" --n-envs 16 \
  --out "runs/loop/arm-$arm/seed-$seed"
