# ADR 0001 — Cloud-agnostic over managed ML platforms

**Status:** Accepted
**Date:** 2026-W19

## Context

The project must run on Kubernetes and demonstrate MLOps competence for hiring. Two paths:

1. **Managed cloud ML** (SageMaker / Vertex AI / Azure ML) — fast to set up, but locks in to one provider and hides the primitives.
2. **Open-source MLOps on K8s** (MLflow, Feast, Argo, KServe/BentoML) — more wiring, but portable and demonstrates the underlying skills.

## Decision

We choose path 2: **cloud-agnostic open-source MLOps on Kubernetes.** Terraform provisions the cluster on either AWS (EKS) or Azure (AKS); the application layer is identical on both.

## Consequences

**Positive:**

- Portfolio demonstrates the *primitives* (model registry, feature store, training orchestration) rather than which buttons to click in a console.
- Project can be reproduced by anyone with a cluster — important for an open-source project.
- Skills transfer between cloud providers; not betting on one job market.

**Negative:**

- More setup work. Authentication, networking, and storage backends all need explicit configuration.
- Some best-in-class managed features (e.g., SageMaker distributed training) are not available without writing them ourselves.

**Mitigation:** Use BentoML in v1 for serving (great DX), then migrate to KServe in v2 to demonstrate the K8s-native serving path. The migration itself is a portfolio artifact.
