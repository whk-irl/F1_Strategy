# Tire degradation model — CPU-only training image.
# Base: slim Python 3.11 to keep the image small.
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv for fast dependency resolution.
RUN pip install --no-cache-dir uv

# Copy dependency manifests first for layer caching.
COPY pyproject.toml .
# Install only the runtime deps (no dev extras).
RUN uv pip install --system --no-cache ".[default]" || \
    uv pip install --system --no-cache \
        fastf1 pandas pyarrow numpy pandera pydantic pydantic-settings \
        prefect boto3 s3fs fsspec mlflow lightgbm scikit-learn scipy tqdm rich typer

# ---------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source.
COPY services/ services/
COPY ml/ ml/

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

CMD ["python", "-m", "ml.models.tire_degradation.train"]
