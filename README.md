# Pitwall AI

> An open-source F1 race strategy engine — pit timing, tire compound selection, and safety-car response — powered by reinforcement learning and a cloud-native MLOps stack.

[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**[🏎️ Live demo →](https://f1strategy-forza.streamlit.app/)**

---

## What it does

Given the current race state, Pitwall AI recommends:

- **When to pit** — balancing track position, tyre life, and pit-loss time
- **Which compound to fit** — soft / medium / hard, with wet-weather overrides
- **How to react to safety cars** — identifying free-stop windows in real time

The frontend opens in **Race Center**:

- **Recorded replay** — watch cars move around the actual circuit trace, with a timing tower,
  selected-driver telemetry, tyre stints, weather and race-control messages. Pause, seek,
  change speed, follow a car or choose another race. OpenF1 location coverage begins in 2023.
- **Live timing** — polls the active OpenF1 session every 15 seconds and displays feed delay.
  Current live data requires an OpenF1 subscription and a server-side `OPENF1_TOKEN`.
  When no session is active, the screen offers recorded replay.
- **Strategy Lab** — retains the existing model comparison and simulated lap explorer.
  The race-center strategy desk uses a labelled nominal dry-race dynamic program.
  The newly trained candidate remains experimental because it did not beat the DP benchmark.

See the [race-center guide](docs/guides/race-center.md) for controls and data semantics.

![Race Center with recorded car positions and race telemetry](docs/assets/race-center.png)

---

## How it works

```
FastF1 / OpenF1
      │
      ▼
Bronze → Silver → Gold  (Parquet on S3, partitioned by year/round/session)
      │
      ├─► Tire degradation model   (LightGBM)
      ├─► Safety-car probability   (LightGBM)
      └─► Race Simulator           (Gymnasium env)
                │
                ▼
          Historical action pre-training → PPO → recent-season fine-tuning
                │
                ▼
          Streamlit Race Center + Strategy Lab
```

The simulator replays recorded field lap times so the agent competes against a realistic grid. The policy is trained with shaped rewards for position gains, tyre-cliff avoidance, and safety-car opportunism.

---

## Tech stack

| Layer | Tool |
|---|---|
| Data | FastF1, OpenF1 API, Parquet, S3 |
| Orchestration | Prefect (dev), Argo Workflows (prod) |
| Experiment tracking | MLflow |
| ML | LightGBM, PyTorch + Stable-Baselines3 (PPO / DQN+PER) |
| Serving | BentoML → FastAPI |
| Frontend | Streamlit |
| Infrastructure | Kubernetes, Helm, Argo CD, Terraform (EKS / AKS) |
| CI/CD | GitHub Actions |
| Observability | Prometheus, Grafana, Loki, Evidently |

---

## Run locally

```bash
git clone https://github.com/whk-irl/F1_Strategy
cd F1_Strategy

# Install dependencies
uv sync

# Copy env config
cp .env.example .env

# Launch the race center (fetches and caches recorded GPS on first use)
uv run streamlit run services/frontend/app.py
```

The repo ships with pre-baked models (`models_baked/`) and race data
(`data/gold_2022_2025.parquet`) for Strategy Lab. Race Center fetches recorded GPS and
telemetry from OpenF1 on first use; subsequent loads use `data/race_center/`.
The hosted demo may still run an earlier release until this branch is deployed.

## Retrain end to end

```bash
uv sync --python 3.11 --extra dev
uv run python -m services.ingestion.acquire --years 2026 --provider auto
uv run python -m ml.pipelines.train --config configs/training.local.json
```

The new workflow acquires completed races with an OpenF1 fallback, rebuilds causal
features, pretrains and fine-tunes the supporting models, validates the simulator,
then runs historical-action pretraining → history-aware PPO → newer-event fine-tuning.
It reserves chronological validation/test events and benchmarks against dynamic
programming on paired episodes. Checkpoints, reports, provenance and MLflow records
are saved locally. Use `--resume` to verify and reuse completed stages.

Read the [training guide](docs/guides/staged-training.md) and
[algorithm decision](docs/decisions/0005-staged-policy-training.md) before changing splits
or using the new 96-input candidate with the legacy 21-input frontend.

The [October 15 retraining result](docs/model-cards/2026-10-15-retraining.md) records
the completed run, including the stronger dynamic-programming baseline and the
unpromoted PPO candidate.

---

## Why I built this

A tifosi dreamed of another Ferrari WDC.

---

## License

MIT
