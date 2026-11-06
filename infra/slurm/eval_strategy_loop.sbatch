#!/usr/bin/env bash
# Score each arm's 5-seed ensemble on validation and stress (never the test split).
#SBATCH --job-name=f1-loop-eval
#SBATCH --partition=defq
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --no-requeue
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the bundle root}"
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PY="${PYTHON:?export PYTHON}"
ARMS=(A B C D)
arm="${ARMS[${SLURM_ARRAY_TASK_ID:?}]}"
cand="runs/loop/arm-$arm"
n=$(ls "$cand"/seed-*/policy.zip 2>/dev/null | wc -l)
echo "arm $arm: $n trained seeds"
[ "$n" -ge 1 ] || { echo "no trained seeds for arm $arm"; exit 1; }
"$PY" -m ml.strategy.protocol evaluate --candidate "$cand" --split val
"$PY" -m ml.strategy.protocol evaluate --candidate "$cand" --split stress --replays 3
