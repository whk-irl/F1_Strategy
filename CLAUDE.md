# CLAUDE.md

This file gives Claude Code the context it needs to work effectively on this repo. Read it first; refer back to it when scope or conventions are unclear.

## What this project is

Pitwall AI is an open-source F1 race strategy engine. Given live race state, it recommends:
- When to pit
- Which tire compound (soft / medium / hard)
- How to react to safety cars / VSCs

The architecture combines a learned race simulator (Gym environment) with a reinforcement-learning policy (PPO), trained on historical F1 data from FastF1 (2018+).

This is a portfolio / research project. **Not** a real-time live timing tool — it simulates "live" decisions over recorded race state. See `docs/scope.md` for explicit non-goals.

## Target audience for the codebase

- Hiring managers and engineers reviewing this as a portfolio piece for ML / MLOps / Data Engineering / Applied ML roles.
- Open-source contributors who want to extend it.

This means: code quality, documentation, ADRs, and tests matter as much as the ML results. Don't take shortcuts that look amateurish (commented-out code, missing type hints, undocumented magic numbers, untested critical paths).

## Tech stack (locked decisions — see `docs/decisions/` for rationale)

| Layer | Tool |
|---|---|
| Language | Python 3.11+ |
| Package manager | uv (preferred) or poetry |
| Data source | FastF1 (primary), Ergast historical CSVs (backup) |
| Storage | Parquet on MinIO (dev) / S3-compatible (prod) |
| Orchestration | Prefect (dev), Argo Workflows (prod K8s) |
| Feature store | Feast |
| Experiment tracking | MLflow (also model registry) |
| Tabular ML | LightGBM |
| Deep learning / RL | PyTorch + Stable-Baselines3 |
| Serving | BentoML (v1) → KServe (v2) |
| API framework | FastAPI |
| Frontend | Streamlit (v1), Next.js (v2) |
| Container orchestration | Kubernetes (kind locally, EKS/AKS in prod) |
| IaC | Terraform |
| Deployment | Helm + Argo CD (GitOps) |
| CI/CD | GitHub Actions |
| Observability | Prometheus, Grafana, Loki, Evidently |
| Data quality | pandera (preferred) or Great Expectations |

**Cloud-agnostic is non-negotiable.** Don't introduce AWS-only or Azure-only dependencies in the application layer. Cloud-specific code lives in `infra/terraform/aws/` and `infra/terraform/azure/` only.

## Repository layout

```
pitwall-ai/
├── docs/                       # architecture, scope, ADRs, model cards
├── infra/
│   ├── terraform/              # cloud-agnostic cluster module + AWS/Azure backends
│   └── helm/                   # one chart per service + a 'platform' chart for shared infra
├── services/
│   ├── ingestion/              # FastF1 → bronze/silver/gold Parquet
│   ├── api/                    # BentoML / FastAPI inference service
│   ├── simulator/              # Gym env + parallel rollout workers
│   └── frontend/               # Streamlit demo app
├── ml/
│   ├── features/               # Feast feature definitions + transforms
│   ├── models/                 # one folder per model: tire_degradation, safety_car, strategy_policy
│   ├── pipelines/              # Argo / Kubeflow training pipelines
│   └── notebooks/              # exploration only — NOT the source of truth
└── tests/
    ├── unit/
    ├── integration/
    └── data_quality/           # pandera schemas, contract tests
```

## Conventions

### Code style
- Format: `ruff format` (Black-compatible). Lint: `ruff check`. Types: `mypy --strict` on `services/` and `ml/`.
- All public functions get type hints and a docstring (Google style).
- Pydantic v2 for all config and API schemas. No raw dicts crossing module boundaries.
- No comments explaining *what* the code does — only *why*, when non-obvious.

### Testing
- `pytest` for everything. `pytest-cov` enforces ≥80% coverage on `services/` and `ml/`.
- Fixtures for race data live in `tests/fixtures/` as small Parquet files (a few laps from one race).
- Data quality tests live in `tests/data_quality/` and run as part of CI.
- Integration tests assume MinIO + MLflow are up via `docker-compose`.

### Notebooks
- Notebooks are for exploration only. **The source of truth is always code in `ml/` or `services/`.**
- Strip outputs before commit (`nbstripout` is in pre-commit hooks).
- If a notebook produces something useful, port the logic into a module before relying on it.

### Git
- Conventional Commits: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`, `ci:`, `perf:`.
- One logical change per commit. Squash-merge PRs.
- Branch naming: `<type>/<short-description>` e.g. `feat/tire-degradation-model`.

### Architecture decisions
- Any non-trivial technical decision (new dependency, deployment change, schema migration) gets an ADR in `docs/decisions/`.
- ADR template is in `docs/decisions/0001-cloud-agnostic-mlops.md`. Number sequentially.

## How data flows

```
FastF1 API
    │
    ▼  (services/ingestion)
Bronze (raw FastF1 dump, Parquet)
    │
    ▼  (cleaning, schema enforcement via pandera)
Silver (per-lap canonical records)
    │
    ▼  (feature engineering)
Gold (ML-ready feature tables, registered in Feast)
    │
    ├──► Tire degradation model (LightGBM)
    ├──► Safety-car probability model (LightGBM)
    └──► Baseline pace model (LightGBM)
              │
              ▼
        Race Simulator (Gym env)
              │
              ▼
        PPO Strategy Policy
              │
              ▼
        BentoML serving → FastAPI → Streamlit frontend
```

**Critical guardrail:** the simulator must achieve Spearman ρ ≥ 0.7 on held-out 2024 race outcomes (with real driver actions replayed) **before** any RL training begins. If validation fails, fix the simulator — never train RL against a broken environment.

## What NOT to do

- ❌ Don't add an LLM agent layer for "natural-language strategy queries." It dilutes the project's story.
- ❌ Don't introduce SageMaker, Vertex AI, or Azure ML SDK calls. Cloud-agnostic means cloud-agnostic.
- ❌ Don't commit raw race data, FastF1 cache, or model artifacts. See `.gitignore`.
- ❌ Don't commit `.tfstate`, `.tfvars`, or kubeconfigs. Ever.
- ❌ Don't write code in notebooks that doesn't have a module equivalent.
- ❌ Don't bypass pandera schemas at silver/gold layer boundaries.
- ❌ Don't add multi-car (team) strategy in v1 — it's an explicit non-goal in `scope.md`.

## Current phase

See README "Roadmap" section. Update the checkboxes as phases complete.

## When making changes, prefer

1. Small, reviewable PRs over large omnibus changes.
2. Adding a test before the fix when reproducing a bug.
3. Updating `CLAUDE.md` when a convention or decision changes — this file should stay current.
4. Writing an ADR for any decision that future-you (or a reviewer) might question.
