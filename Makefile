.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Local dev variables (override on CLI)
# ---------------------------------------------------------------------------
SEASON  ?= 2024
ROUND   ?= 1
SESSION ?= R

# ---------------------------------------------------------------------------
# Cloud variables — set in your shell or .env before running cloud targets.
# These are NOT read from PITWALL_ env vars; they're CI/Makefile-level.
# ---------------------------------------------------------------------------
AWS_ACCOUNT_ID  ?= $(shell aws sts get-caller-identity --query Account --output text 2>/dev/null)
AWS_REGION      ?= us-east-1
ECR_REGISTRY    ?= $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
IMAGE_TAG       ?= latest
EKS_CLUSTER     ?= pitwall-prod
ARGO_SERVER     ?= localhost:2746   # port-forward: kubectl port-forward svc/argo-server 2746 -n argo
TRAINING_SEASONS ?= 2021,2022,2023,2024
TOTAL_TIMESTEPS  ?= 1000000

.PHONY: help setup up down ingest ingest-season \
        train-tire train-safety-car validate-simulator train-policy demo \
        ecr-login build-images push-images \
        submit-train-tire submit-train-safety-car submit-train-policy \
        kubeconfig-update \
        test test-unit test-dq lint format typecheck ci clean

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
# Local training (no GPU, small dataset — for development iteration)
# ---------------------------------------------------------------------------
train-tire:  ## Train tire degradation model locally (CPU)
	uv run python -m ml.models.tire_degradation.train

train-safety-car:  ## Train safety-car model locally (CPU)
	uv run python -m ml.models.safety_car.train

validate-simulator:  ## Replay 2024 races through tire model; need Spearman ρ ≥ 0.7
	uv run python -m services.simulator.validator --seasons $(SEASON)

train-policy:  ## Train PPO policy locally (slow — CPU only)
	uv run python -m ml.models.strategy_policy.train

# ---------------------------------------------------------------------------
# Cloud training — AWS S3 data + EKS GPU nodes via Argo Workflows
# Requires: AWS credentials, EKS kubeconfig, ECR access.
# ---------------------------------------------------------------------------
ecr-login:  ## Authenticate Docker to ECR
	aws ecr get-login-password --region $(AWS_REGION) | \
		docker login --username AWS --password-stdin $(ECR_REGISTRY)

kubeconfig-update:  ## Update local kubeconfig for EKS cluster
	aws eks update-kubeconfig --name $(EKS_CLUSTER) --region $(AWS_REGION)

build-images:  ## Build all training Docker images locally
	docker build -t $(ECR_REGISTRY)/pitwall-tire-degradation:$(IMAGE_TAG) \
		-f ml/models/tire_degradation/Dockerfile .
	docker build -t $(ECR_REGISTRY)/pitwall-safety-car:$(IMAGE_TAG) \
		-f ml/models/safety_car/Dockerfile .
	docker build -t $(ECR_REGISTRY)/pitwall-strategy-policy:$(IMAGE_TAG) \
		-f ml/models/strategy_policy/Dockerfile .

push-images: ecr-login build-images  ## Build + push all training images to ECR
	docker push $(ECR_REGISTRY)/pitwall-tire-degradation:$(IMAGE_TAG)
	docker push $(ECR_REGISTRY)/pitwall-safety-car:$(IMAGE_TAG)
	docker push $(ECR_REGISTRY)/pitwall-strategy-policy:$(IMAGE_TAG)

submit-train-tire: kubeconfig-update  ## Submit tire degradation training to EKS (Argo)
	argo submit --from workflowtemplate/train-tire-degradation \
		-p image=$(ECR_REGISTRY)/pitwall-tire-degradation:$(IMAGE_TAG) \
		-p training_seasons=$(TRAINING_SEASONS) \
		--server $(ARGO_SERVER) --watch

submit-train-safety-car: kubeconfig-update  ## Submit safety-car training to EKS (Argo)
	argo submit --from workflowtemplate/train-safety-car \
		-p image=$(ECR_REGISTRY)/pitwall-safety-car:$(IMAGE_TAG) \
		-p training_seasons=$(TRAINING_SEASONS) \
		--server $(ARGO_SERVER) --watch

submit-train-policy: kubeconfig-update  ## Submit GPU strategy policy training to EKS (Argo)
	argo submit --from workflowtemplate/train-strategy-policy \
		-p image=$(ECR_REGISTRY)/pitwall-strategy-policy:$(IMAGE_TAG) \
		-p total_timesteps=$(TOTAL_TIMESTEPS) \
		--server $(ARGO_SERVER) --watch

# ---------------------------------------------------------------------------
# Terraform — AWS infrastructure
# Phase 1: S3 + ECR (run this first to start ingesting data)
# Phase 2: VPC + EKS + IRSA (run when ready to train on GPU)
# ---------------------------------------------------------------------------
TF_DIR        := infra/terraform/aws
TF_BOOTSTRAP  := infra/terraform/aws/bootstrap

tf-bootstrap:  ## (One-time) Create S3 state bucket + DynamoDB lock table
	terraform -chdir=$(TF_BOOTSTRAP) init
	terraform -chdir=$(TF_BOOTSTRAP) apply -auto-approve

tf-init:  ## Initialise Terraform with remote S3 backend (run after tf-bootstrap)
	terraform -chdir=$(TF_DIR) init

tf-plan-phase1:  ## Preview Phase 1 changes (S3 + ECR only)
	terraform -chdir=$(TF_DIR) plan \
		-target=aws_s3_bucket.data_lake \
		-target=aws_s3_bucket_versioning.data_lake \
		-target=aws_s3_bucket_server_side_encryption_configuration.data_lake \
		-target=aws_s3_bucket_public_access_block.data_lake \
		-target=aws_s3_bucket_lifecycle_configuration.data_lake \
		-target=aws_ecr_repository.training \
		-target=aws_ecr_lifecycle_policy.training

tf-apply-phase1:  ## Apply Phase 1 (S3 + ECR) — needed before first ingestion run
	terraform -chdir=$(TF_DIR) apply \
		-target=aws_s3_bucket.data_lake \
		-target=aws_s3_bucket_versioning.data_lake \
		-target=aws_s3_bucket_server_side_encryption_configuration.data_lake \
		-target=aws_s3_bucket_public_access_block.data_lake \
		-target=aws_s3_bucket_lifecycle_configuration.data_lake \
		-target=aws_ecr_repository.training \
		-target=aws_ecr_lifecycle_policy.training

tf-apply-phase2:  ## Apply Phase 2 (VPC + EKS + IRSA) — needed for GPU training
	terraform -chdir=$(TF_DIR) apply

tf-output:  ## Show Terraform outputs (bucket name, ECR registry, kubeconfig cmd)
	terraform -chdir=$(TF_DIR) output

tf-destroy-eks:  ## Destroy EKS + VPC only (keeps S3 data intact)
	terraform -chdir=$(TF_DIR) destroy \
		-target=module.eks \
		-target=module.vpc \
		-target=kubernetes_namespace.pitwall \
		-target=kubernetes_service_account.training \
		-target=kubernetes_service_account.ingestion

tf-destroy:  ## DANGER: destroy all cloud resources including S3 data
	terraform -chdir=$(TF_DIR) destroy

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
