.DEFAULT_GOAL := help

# Configurable via CLI: `make ingest SEASON=2023 ROUND=5`
SEASON  ?= 2024
ROUND   ?= 1
SESSION ?= R

.PHONY: help setup up down ingest train-tire train-safety-car train-policy demo \
        test lint typecheck ci clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
setup:  ## Install all deps (uv) and install pre-commit hooks
	uv sync --all-extras
	pre-commit install
	pre-commit install --hook-type commit-msg

# ---------------------------------------------------------------------------
# Local stack
# ---------------------------------------------------------------------------
up:  ## Start local dev stack (MinIO, PostgreSQL, MLflow, Prefect)
	docker compose up -d
	@echo "MinIO console  → http://localhost:9001  (minioadmin / minioadmin)"
	@echo "MLflow UI      → http://localhost:5000"
	@echo "Prefect UI     → http://localhost:4200"

down:  ## Tear down local dev stack
	docker compose down

# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
ingest:  ## Ingest one race session into bronze/silver/gold (SEASON ROUND SESSION)
	uv run python -m services.ingestion.pipeline \
		--year $(SEASON) --round $(ROUND) --session $(SESSION)

ingest-season:  ## Ingest all races for a full season (SEASON)
	uv run python -m services.ingestion.pipeline \
		--year $(SEASON) --all-rounds

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
train-tire:  ## Train tire degradation LightGBM model
	uv run python -m ml.models.tire_degradation.train

train-safety-car:  ## Train safety-car probability LightGBM model
	uv run python -m ml.models.safety_car.train

train-policy:  ## Train PPO strategy policy (slow — ~30 min on CPU)
	uv run python -m ml.models.strategy_policy.train

# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
demo:  ## Launch Streamlit demo (http://localhost:8501)
	uv run streamlit run services/frontend/app.py

# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------
test:  ## Run full test suite with coverage (requires local stack)
	uv run pytest

test-unit:  ## Run only unit tests (no external services required)
	uv run pytest tests/unit -v

test-dq:  ## Run data-quality contract tests
	uv run pytest tests/data_quality -v

lint:  ## Lint with ruff and check formatting
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-fix lint issues and format code
	uv run ruff check --fix .
	uv run ruff format .

typecheck:  ## Run mypy strict type checking on services/ and ml/
	uv run mypy services/ ml/

ci:  ## Run the full CI pipeline locally (lint → typecheck → test)
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) test-unit
	$(MAKE) test-dq

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------
clean:  ## Remove Python caches, coverage reports, and build artefacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"   -exec rm -rf {} + 2>/dev/null || true
	rm -f coverage.xml .coverage
