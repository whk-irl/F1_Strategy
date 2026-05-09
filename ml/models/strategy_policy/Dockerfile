# PPO strategy policy — GPU training image.
#
# Base: PyTorch official CUDA image.  The CUDA / cuDNN version must match the
# driver on the EKS GPU node group (configured in infra/terraform/aws/).
#
# CUDA 12.1 is compatible with AWS GPU instances running NVIDIA driver ≥ 525.
ARG PYTORCH_VERSION=2.2.0
ARG CUDA_VERSION=12.1
ARG CUDNN_VERSION=8

FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime AS builder

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
# Install all runtime deps (torch is already in the base image).
RUN uv pip install --system --no-cache \
        fastf1 pandas pyarrow numpy pandera pydantic pydantic-settings \
        prefect boto3 s3fs fsspec mlflow \
        stable-baselines3 gymnasium scipy scikit-learn tqdm rich typer

FROM pytorch/pytorch:${PYTORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime

WORKDIR /app

COPY --from=builder /opt/conda/lib/python3.11/site-packages \
                    /opt/conda/lib/python3.11/site-packages
COPY --from=builder /opt/conda/bin /opt/conda/bin

COPY services/ services/
COPY ml/ ml/

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# /dev/shm is extended at runtime via an emptyDir volume (see Argo workflow).
CMD ["python", "-m", "ml.models.strategy_policy.train"]
