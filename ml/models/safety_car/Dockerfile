# Safety-car probability model — CPU-only training image.
FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
RUN uv pip install --system --no-cache \
        fastf1 pandas pyarrow numpy pandera pydantic pydantic-settings \
        prefect boto3 s3fs fsspec mlflow lightgbm scikit-learn scipy tqdm rich typer

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY services/ services/
COPY ml/ ml/

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

CMD ["python", "-m", "ml.models.safety_car.train"]
