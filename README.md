# Pitwall AI

> An open-source F1 race strategy engine — pit timing, tire compound choice, and safety-car response — built on a cloud-agnostic Kubernetes + open-source MLOps stack.

[![CI](https://img.shields.io/badge/ci-pending-lightgrey)](.) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## What this is

Pitwall AI recommends optimal in-race strategy decisions for a Formula 1 team:

- **When to pit** the car
- **Which tire compound** to fit (soft / medium / hard)
- **How to react** to safety cars and virtual safety cars

It does this by combining a learned race simulator with a reinforcement-learning policy, trained on historical F1 data from 2018 onward via [FastF1](https://docs.fastf1.dev/).

> ⚠️ This is a research / portfolio project. It does not consume live F1 timing — it simulates live decisions over historical race state. See [`docs/scope.md`](docs/scope.md) for what is and isn't real-time.

## Architecture

```
                 ┌──────────────────────────┐
                 │   FastF1 / Ergast        │
                 │   (open F1 data)         │
                 └────────────┬─────────────┘
                              │  Prefect / Argo CronJob
                              ▼
                 ┌──────────────────────────┐
                 │  Bronze → Silver → Gold  │
                 │  Parquet on S3 / MinIO   │
                 └────────────┬─────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐    ┌──────────┐
        │  Feast   │   │  MLflow  │    │  Argo    │
        │ Features │   │ Tracking │    │ Workflow │
        └────┬─────┘   └────┬─────┘    └────┬─────┘
             │              │               │
             └──────┬───────┴───────────────┘
                    ▼
        ┌────────────────────────────┐
        │  Race Simulator (Gym env)  │
        │  + Tire / SC / Pace models │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  PPO Strategy Policy (RL)  │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  BentoML / FastAPI service │
        │  + Streamlit frontend      │
        └────────────────────────────┘

Everything deploys to Kubernetes (kind / EKS / AKS) via Helm + Argo CD.
Observability: Prometheus, Grafana, Loki, Evidently.
```

See [`docs/architecture.md`](docs/architecture.md) for the full diagram and component-by-component rationale.

## Tech stack

| Concern | Tool |
|---|---|
| Data | FastF1, Parquet, MinIO/S3 |
| Orchestration | Prefect (dev), Argo Workflows (prod) |
| Feature store | Feast |
| Experiment tracking | MLflow |
| ML | LightGBM (tabular), PyTorch + Stable-Baselines3 (RL) |
| Serving | BentoML → FastAPI |
| Frontend | Streamlit (v1), Next.js (v2) |
| Orchestration runtime | Kubernetes (kind / EKS / AKS via Terraform) |
| Deployment | Helm + Argo CD (GitOps) |
| CI/CD | GitHub Actions |
| Observability | Prometheus, Grafana, Loki, Evidently |
| IaC | Terraform |

## Repository layout

```
pitwall-ai/
├── README.md
├── LICENSE
├── pyproject.toml
├── Makefile
├── docker-compose.yml          # local end-to-end stack
├── .github/workflows/          # CI/CD pipelines
├── .pre-commit-config.yaml
│
├── docs/
│   ├── architecture.md
│   ├── scope.md                # explicit non-goals
│   ├── decisions/              # ADRs (architecture decision records)
│   └── model-cards/
│
├── infra/
│   ├── terraform/
│   │   ├── modules/cluster/    # cloud-agnostic K8s
│   │   ├── aws/                # EKS backend
│   │   └── azure/              # AKS backend
│   └── helm/
│       ├── pitwall-api/
│       ├── pitwall-frontend/
│       ├── pitwall-ingestion/
│       └── platform/           # mlflow, minio, monitoring
│
├── services/
│   ├── ingestion/              # FastF1 → bronze/silver/gold
│   ├── api/                    # BentoML / FastAPI inference
│   ├── simulator/              # Gym env + workers
│   └── frontend/               # Streamlit app
│
├── ml/
│   ├── features/               # Feast definitions + transforms
│   ├── models/
│   │   ├── tire_degradation/
│   │   ├── safety_car/
│   │   └── strategy_policy/    # PPO agent
│   ├── pipelines/              # Argo / Kubeflow pipelines
│   └── notebooks/              # exploration only
│
└── tests/
    ├── unit/
    ├── integration/
    └── data_quality/           # Great Expectations / pandera suites
```

## Quickstart (local)

```bash
# 1. Clone & install
git clone https://github.com/<you>/pitwall-ai
cd pitwall-ai
make setup

# 2. Bring up local stack (MinIO, MLflow, Postgres, Prefect)
make up

# 3. Ingest a season of data
make ingest SEASON=2024

# 4. Train the supporting models
make train-tire
make train-safety-car

# 5. Train the RL policy (slow; ~30 min on CPU)
make train-policy

# 6. Launch the demo frontend
make demo
# → http://localhost:8501
```


## Why I built this

A tifosi dreamed of another Ferrari WDC.

## License

MIT
