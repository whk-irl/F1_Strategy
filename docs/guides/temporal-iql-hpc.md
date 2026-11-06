# Temporal IQL on an HPC cluster

The job performs acquisition, measured pit-time fitting, supporting-model
pretraining and adaptation, simulator validation, IQL source pretraining, recent
event fine-tuning, and one final held-out evaluation.

Create the environment and verify a short run before submission:

```bash
uv sync --frozen
uv run pytest tests/unit/test_temporal_iql.py tests/unit/ingestion/test_acquisition.py
uv run python -m ml.pipelines.train_iql --config configs/training.iql-smoke.json
```

For SLURM, copy the repository to cluster storage and edit only the `#SBATCH`
account, partition, and GPU directives required by the site. Then submit:

```bash
sbatch infra/slurm/train_temporal_iql.sbatch
```

The resumable acquisition cache lives in `.fastf1_cache`; validated Parquet
partitions and checksums live in `data/lake`. Keep both on persistent storage.
Training artifacts are written to `runs/temporal-iql-2021-2026`, including the
two checkpoints, per-epoch validation history, exact source hashes, paired test
episodes, bootstrap comparison, and a promotion flag.

For multiple seeds, copy the JSON config per seed and give each an independent
`output_dir`, or submit the included five-seed array after acquisition completes:

```bash
sbatch infra/slurm/train_temporal_iql_array.sbatch
```

Do not select a checkpoint or hyperparameter from the final test results.
