# Pitwall AI

> An open-source F1 race strategy engine — pit timing, tire compound selection, and safety-car response — powered by reinforcement learning and a cloud-native MLOps stack.

[![CI](https://github.com/whk-irl/F1_Strategy/actions/workflows/ci.yml/badge.svg)](https://github.com/whk-irl/F1_Strategy/actions) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**[Live demo →](https://f1strategy-forza.streamlit.app/)**

---

## What it does

Given the current race state, Pitwall AI recommends:

- **When to pit** — balancing track position, tyre life, and pit-loss time
- **Which compound to fit** — soft / medium / hard, with wet-weather overrides
- **How to react to safety cars** — identifying free-stop windows in real time

Two modes:

- **Race Replay** — step through any 2022–2025 race lap by lap. The red line shows where the driver would have finished under Pitwall AI's strategy; the grey line shows what the team actually did.
- **Live Race** — connects to the [OpenF1 public API](https://openf1.org/) during a race weekend and streams real-time recommendations.

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
          PPO Strategy Policy  (Stable-Baselines3)
                │
                ▼
          Streamlit frontend   (Race Replay + Live Race tabs)
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
cd F1_Strategy/F1_Strategy

# Install dependencies
uv sync

# Copy env config
cp .env.example .env

# Launch the demo (uses baked models + pre-built data)
streamlit run services/frontend/app.py
```

The repo ships with pre-baked models (`models_baked/`) and race data (`data/gold_2022_2025.parquet`) so the frontend runs without any training step.

---

## Why I built this

A tifosi dreamed of another Ferrari WDC.

---

## License

MIT
