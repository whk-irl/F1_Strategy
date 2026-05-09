# ADR 0002 — AWS S3 for cloud storage + EKS GPU nodes for model training

**Status:** Accepted
**Date:** 2026-W19

## Context

The project needs:
1. A durable, scalable store for bronze/silver/gold Parquet datasets that survives beyond the local dev machine.
2. Sufficient GPU compute to train the PPO strategy policy (Stable-Baselines3 + PyTorch) in a reasonable time (~30 min on a single GPU vs ~8 h on CPU).

The architecture is cloud-agnostic at the application layer, but a single concrete cloud backend must be chosen for the initial portfolio deployment.

## Decision

**Storage:** AWS S3.

- Application code uses the existing boto3 S3 client with `endpoint_url=None` (real S3) or `endpoint_url=<MinIO>` (local dev), controlled by `PITWALL_STORAGE_BACKEND`.
- The S3 bucket (`pitwall-ai-prod`) is created and configured by Terraform (`infra/terraform/aws/`).  The application never auto-creates S3 buckets to preserve the IaC audit trail.
- Credentials flow exclusively via IRSA (IAM Roles for Service Accounts) on EKS — no access keys stored in application config.

**GPU compute:** Argo Workflows submitted to an EKS GPU node group.

- Three `WorkflowTemplate` manifests live in `infra/argo/workflows/`.  They are cluster-level resources submitted via `argo submit --from workflowtemplate/<name>` or the GitHub Actions `train-cloud.yml` workflow.
- CPU models (tire degradation, safety-car) run on general-purpose nodes. The PPO policy runs on a dedicated GPU node group (`eks.amazonaws.com/nodegroup: gpu-training`) using `nvidia.com/gpu: 1` resource requests and the appropriate toleration.
- `/dev/shm` is extended at runtime via a Memory-backed `emptyDir` volume to prevent PyTorch multiprocessing from crashing on shared-memory limits.
- CUDA version (12.1) and driver requirement (≥ 525) are pinned in the `strategy_policy/Dockerfile` and must match the EKS node AMI.

## Consequences

**Positive:**

- Dataset durability: S3 object storage is independent of the training cluster lifecycle.
- Reproducibility: Parquet files are immutable, Hive-partitioned objects — any training run can be re-launched against the same data.
- Cost control: GPU nodes can be zero-scaled (min=0 in the EKS managed node group) and only spin up when a WorkflowTemplate is submitted.
- Cloud-agnostic application layer is preserved: switching to Azure requires only new Terraform modules and a `storage_backend=azure` config variant — application source code is unchanged.

**Negative:**

- S3 bucket provisioning is a Terraform prerequisite; new environments need `terraform apply` before the first ingestion run.
- EKS GPU node group adds ~$2–4/h when active (g4dn.xlarge); must be zero-scaled after training to control cost.
- IRSA setup requires an OIDC provider on the EKS cluster — one-time Terraform configuration.

**Mitigation:**

- Terraform workspace + `make up-cloud` automates the full EKS + IRSA setup.
- A GitHub Actions `train-cloud.yml` workflow (manual trigger + `workflow_dispatch`) ensures GPU nodes are only used intentionally, not on every push.
