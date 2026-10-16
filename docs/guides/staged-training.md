# Acquisition through pretraining and fine-tuning

Use Python 3.11 and the committed `uv.lock`. CUDA is used for the sequence model when
available; PPO's small MLP runs on CPU, consistent with the
[Stable-Baselines3 guidance](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).

```powershell
uv sync --python 3.11 --extra dev
uv run python -m services.ingestion.acquire --years 2026 --provider auto --as-of 2026-10-16T00:00:00+00:00
uv run python -m ml.pipelines.train --config configs/training.local.json
```

If FastF1's timing feed is unavailable, choose `--provider openf1` directly. OpenF1's
[historical API](https://openf1.org/docs/) is available from 2023 onward. The supplied
2022–2025 gold table seeds older history; all derived features are rebuilt with the
new causal definitions. Acquisition replaces whole matching legacy sessions.

The cutoff above represents the evening of October 15 in America/Chicago. Only
races whose scheduled start was at least a day earlier are eligible. Future races
and races in progress are not downloaded. The local profile requires enough 2026
events to populate its explicit validation and test periods; missing splits fail.

## Stages

1. **Acquisition:** discover completed rounds, download laps/weather/metadata, validate
   bronze, silver and gold, and atomically save partition files and their checksums.
   OpenF1 raw endpoint responses are cached under `data/lake/openf1_raw`.
2. **Preparation:** rebuild legacy features and validate unique race/driver/lap keys.
   Save exact source provenance and event membership.
3. **World models:** pretrain LightGBM tire and safety-car models on source events;
   continue boosting on the adaptation events, with validation early stopping.
4. **Sequence model:** train TCN/GRU next-lap prediction, then transfer its weights
   to adaptation events. Normalization stays fixed to the source training data.
5. **Simulator gates:** replay real actions with both validators. Policy learning is
   blocked if either mean Spearman correlation is below 0.7.
6. **Behavioral pretraining:** imitate historical actions, with held-out source-event
   early stopping. Pit timing and next-compound labels are separated from observations.
7. **Policy pretraining:** improve the actor and learn the critic with PPO rollouts
   on source training events. Save the best source-validation and final checkpoints.
8. **Fine-tuning:** load the selected pretrained policy and optimizer, lower the
   learning rate, and train on newer events. Save the best recent-validation checkpoint.
9. **Evaluation and packaging:** compare policies and baselines on paired held-out
   episodes, write reports/checksums, and log artifacts to MLflow.

The local profile budgets 200,000 source PPO steps and 100,000 fine-tuning steps.
PPO rounds these budgets up to complete rollout batches. The sequence model has a
40-epoch source budget and 15-epoch fine-tuning budget with early stopping.

## Resume and inspect

```powershell
uv run python -m ml.pipelines.train --config configs/training.local.json --resume
uv run python -m ml.pipelines.train --config configs/training.local.json --resume --through gate
uv run mlflow ui --backend-store-uri sqlite:///runs/mlflow.db
```

Resume operates at completed stage boundaries and verifies all stage artifact hashes.
A failed training stage restarts that stage. A changed configuration, source file,
dataset or model checksum requires a new output directory. A partially written file
never qualifies as a completed artifact.

The saved October 15 run predates the October 16 Race Center source changes. Its
artifacts and provenance are retained, and inference still verifies their checksums.
To train against this newer checkout, copy the configuration and choose a new
`output_dir`; the source-change guard deliberately rejects resuming that older run.

Edit a copy of the JSON config to change epoch/step budgets, cutoffs or output path.
`validation_rounds` applies only to source-year holdouts. The recent-data boundaries
are chronological `(year, round)` pairs, including the year in the split identity.
The 2024 simulator gate remains mandatory. `tracking_uri: null` disables MLflow logging.

Outputs are under the configured `runs/retrain-20261015/` directory:

| Artifact | Purpose |
|---|---|
| `run.json`, `config.json`, `environment.json` | Stage status, exact config, versions and code digest |
| `dataset.json`, `splits.json` | Source hashes and exact event membership |
| `tire/`, `sc/` | Pretrained and fine-tuned LightGBM boosters |
| `tire_tcn_gru/` | Sequence weights, architecture, normalization and loss history |
| `simulator_validation.json` | Both simulator gates with eligible event results |
| `policy/behavior.zip` | Historical-action pretrained policy |
| `policy/pretrained.zip`, `policy/pretrained_last.zip` | Selected and final source PPO checkpoints |
| `policy/finetuned.zip`, `policy/finetuned_last.zip` | Selected and final fine-tuning checkpoints |
| `report.md`, `report.json` | Metrics, paired episodes, uncertainty and limitations |
| `manifest.json` | Bundle schema, paths, checksums and promotion state |

## Inference

```python
from ml.models.strategy_policy.predict_staged import StagedPolicyPredictor

policy = StagedPolicyPredictor("runs/retrain-20261015")
policy.reset()  # once per race/driver, also when seeking backwards in a replay
result = policy.recommend(observation_21, used_compounds={0, 1})
```

Supply observations in the existing simulator's 21-value order, one update per lap.
The adapter appends compound-use flags and maintains four frames. It verifies the
checkpoint digest and input schema. The candidate supports dry actions only. The
legacy Streamlit models have a different schema; the new artifact is delivered as
a separately loadable candidate rather than overwriting those models.

See [ADR 0005](../decisions/0005-staged-policy-training.md) for the research, simulator
assumptions, excluded cases and promotion criteria. Checkpoint quality is determined
by the saved metrics, not by completing the training budget.
